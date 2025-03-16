import os
import getpass
from typing import TypedDict, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import sqlite3

# --- Setup OpenAI API Key (if not already set)---
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

class ClaimsProcessingState(TypedDict):
    """Represents the state of our claims processing analysis."""
    claims_data: Optional[List[Dict]]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    user_prompt: Optional[str]
    agent_response: Optional[str]
    total_claims: Optional[int]
    accepted_claims: Optional[int]
    denied_claims: Optional[int]
    outstanding_claims: Optional[int]
    claim_submission_rate: Optional[float]
    claim_acceptance_rate: Optional[float]
    average_processing_time: Optional[timedelta]
    denial_rate: Optional[float]
    average_payment_amount: Optional[float]
    analysis_summary: str
    messages: List  # Keep track of conversation history

# --- Node Definitions --- (initialize_claims_data, calculate_*, analyze_results remain unchanged)

def initialize_claims_data(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Loads claims data and initializes the state."""
    claims_data = [
        {"claim_id": "C1", "submission_date": datetime(2024, 1, 5), "processing_date": datetime(2024, 1, 10), "status": "Accepted", "payment_amount": 100.0},
        {"claim_id": "C2", "submission_date": datetime(2024, 1, 7), "processing_date": datetime(2024, 1, 15), "status": "Accepted", "payment_amount": 150.0},
        {"claim_id": "C3", "submission_date": datetime(2024, 1, 10), "processing_date": datetime(2024, 1, 12), "status": "Denied", "payment_amount": 0.0},
        {"claim_id": "C4", "submission_date": datetime(2024, 1, 12), "processing_date": None, "status": "Pending", "payment_amount": 0.0},
        {"claim_id": "C5", "submission_date": datetime(2024, 1, 15), "processing_date": datetime(2024, 1, 20), "status": "Accepted", "payment_amount": 120.0},
    ]
    state["claims_data"] = claims_data
    state["start_date"] = datetime(2024, 1, 1)
    state["end_date"] = datetime(2024, 1, 31)
    state["analysis_summary"] = ""
    state["user_prompt"] = state.get("user_prompt", "")  # Initialize from input, or use ""
    state["agent_response"] = ""
    state["messages"] = []
    state["messages"].append(HumanMessage(content="Claims data initialized."))
    return state

def calculate_basic_metrics(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Calculates total, accepted, denied, and outstanding claims."""
    if state["claims_data"] is None:
        state["messages"].append(AIMessage(content="No claims data available."))
        return state
    claims = state["claims_data"]

    state["total_claims"] = len(claims)
    state["accepted_claims"] = sum(1 for claim in claims if claim["status"] == "Accepted")
    state["denied_claims"] = sum(1 for claim in claims if claim["status"] == "Denied")
    state["outstanding_claims"] = sum(1 for claim in claims if claim["status"] == "Pending")

    state["messages"].append(AIMessage(content="Calculated basic claim counts."))
    return state

def calculate_rates(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Calculates submission, acceptance, and denial rates."""
    if state["claims_data"] is None:
        return state
    claims = state["claims_data"]
    total_claims = state.get("total_claims", 0)
    if total_claims > 0:
        state["claim_submission_rate"] = total_claims / (state["end_date"] - state["start_date"]).days
        state["claim_acceptance_rate"] = (state.get("accepted_claims", 0) / total_claims) * 100
        state["denial_rate"] = (state.get("denied_claims", 0) / total_claims) * 100
    state["messages"].append(AIMessage(content="Calculated claim rates."))
    return state

def calculate_processing_time(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Calculates the average processing time."""
    if state["claims_data"] is None:
        return state

    claims = state["claims_data"]
    processing_times = []
    for claim in claims:
        if claim["status"] != "Pending" and claim["processing_date"] is not None:
            processing_time = claim["processing_date"] - claim["submission_date"]
            processing_times.append(processing_time)

    if processing_times:
        total_time = sum(processing_times, timedelta())
        state["average_processing_time"] = total_time / len(processing_times)
    state["messages"].append(AIMessage(content="Calculated average processing time."))
    return state

def calculate_payment_amount(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Calculates the average payment amount."""
    if state["claims_data"] is None:
        return state

    claims = state["claims_data"]
    total_payment = sum(claim["payment_amount"] for claim in claims if claim["payment_amount"] is not None)
    accepted_claims_count = state.get("accepted_claims", 0)
    if accepted_claims_count > 0:
        state["average_payment_amount"] = total_payment / accepted_claims_count
    state["messages"].append(AIMessage(content="Calculated average payment amount."))
    return state

def analyze_results(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Analyzes the metrics and generates a summary."""
    summary = []
    summary.append("Claims Processing Analysis:")
    summary.append(f"  Total Claims: {state.get('total_claims', 'N/A')}")
    summary.append(f"  Accepted Claims: {state.get('accepted_claims', 'N/A')}")
    summary.append(f"  Denied Claims: {state.get('denied_claims', 'N/A')}")
    summary.append(f"  Outstanding Claims: {state.get('outstanding_claims', 'N/A')}")
    summary.append(f"  Claim Submission Rate: {state.get('claim_submission_rate', 'N/A'):.2f} claims/day")
    summary.append(f"  Claim Acceptance Rate: {state.get('claim_acceptance_rate', 'N/A'):.2f}%")
    summary.append(f"  Denial Rate: {state.get('denial_rate', 'N/A'):.2f}%")
    avg_processing_time = state.get('average_processing_time', 'N/A')
    if isinstance(avg_processing_time, timedelta):
        summary.append(f"  Average Processing Time: {avg_processing_time.days} days, {avg_processing_time.seconds // 3600} hours")
    else:
        summary.append(f"  Average Processing Time: {avg_processing_time}")
    summary.append(f"  Average Payment Amount: ${state.get('average_payment_amount', 'N/A'):.2f}")

    state["analysis_summary"] = "\n".join(summary)
    state["messages"].append(HumanMessage(content="Claims processing analysis complete."))
    return state

# --- Agent Node ---

@tool
def get_metric_value(metric_name: str, state:Dict) -> str:
    """Retrieves the value of a calculated metric.

    Args:
        metric_name: The name of the metric (e.g., "denial_rate").
        state: The current state (passed as a dictionary).
    """
    metric_value = state.get(metric_name)
    if metric_value is None:
        return f"Metric '{metric_name}' is not available."
    if isinstance(metric_value, timedelta):
        return f"{metric_value.days} days, {metric_value.seconds // 3600} hours"
    if isinstance(metric_value, float):
        return f"{metric_value:.2f}"
    return str(metric_value)

def agent_node(state: ClaimsProcessingState) -> ClaimsProcessingState:
    """Processes the user prompt using a LangChain agent."""

    prompt_text = state["user_prompt"]
    if not prompt_text:
        state["agent_response"] = "No prompt provided."
        return state

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

    # --- KEY CHANGE: Create a separate function to get the state ---
    def get_state(input_dict: dict) -> Dict:
        """Extract the state from the input dictionary."""
        return input_dict['state']

    # --- KEY CHANGE: Use a lambda function to access the state ---
    tools = [RunnableLambda(get_metric_value).with_config({"run_name": "GetMetricValue"})]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that provides information about insurance claims processing.  "
                "You have access to calculated metrics. Use the 'get_metric_value' tool. "
                "Answer concisely and factually. If you cannot answer, say so.",
            ),
            MessagesPlaceholder(variable_name="messages"), # Use messages
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), #Keep scratchpad
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Set verbose=True for debugging
        return_intermediate_steps=True,
        handle_parsing_errors=True, #Add this to handle parsing error
    )

    # --- KEY CHANGE: Pass messages to the agent ---
    result = agent_executor.invoke({"input": prompt_text, "messages": state["messages"], "state": state})
    state["agent_response"] = result["output"]
    state["messages"].append(AIMessage(content=state["agent_response"]))  # Add to conversation
    return state

# --- Graph Definition ---

def create_workflow():
    """Creates and returns the LangGraph workflow."""
    workflow = StateGraph(ClaimsProcessingState)
    workflow.add_node("initialize_claims_data", initialize_claims_data)
    workflow.add_node("calculate_basic_metrics", calculate_basic_metrics)
    workflow.add_node("calculate_rates", calculate_rates)
    workflow.add_node("calculate_processing_time", calculate_processing_time)
    workflow.add_node("calculate_payment_amount", calculate_payment_amount)
    workflow.add_node("analyze_results", analyze_results)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("initialize_claims_data")
    workflow.add_edge("initialize_claims_data", "calculate_basic_metrics")
    workflow.add_edge("calculate_basic_metrics", "calculate_rates")
    workflow.add_edge("calculate_rates", "calculate_processing_time")
    workflow.add_edge("calculate_processing_time", "calculate_payment_amount")
    workflow.add_edge("calculate_payment_amount", "analyze_results")
    workflow.add_edge("analyze_results", "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

def main():
    """Main function to run the claims processing analysis."""
    app = create_workflow()
    while True:
        user_prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
        inputs = {"user_prompt": user_prompt} # Correctly pass the user prompt
        result = app.invoke(inputs)
        print(f"Agent Response: {result['agent_response']}\n")

if __name__ == "__main__":
    main()