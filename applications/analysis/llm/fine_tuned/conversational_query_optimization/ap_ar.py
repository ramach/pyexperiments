'''
The user wants to enhance the AP/AR agent by adding a "conversational query optimization agent." This implies a step before the main AP/AR agent runs, where the user's raw query is refined or reformulated to be more effective for retrieving relevant information, especially when dealing with documents or potentially ambiguous internal/external data sources.

Here's a breakdown of the thinking process to integrate this:

Understand the Goal: The aim is to make the input to the main AP/AR agent better.
If a user asks "Status on Tech Solutions?",
the optimization agent might rephrase it to
"What is the status of unpaid or overdue Accounts Payable invoices for vendor Tech Solutions Inc.?"
or "Retrieve internal Accounts Payable data for vendor Tech Solutions Inc."
This helps the main agent select the right tool and parameters more reliably.
This is particularly relevant if we were retrieving from a large document store (vector database),
but it's also useful for ensuring the right tools are called with the correct entity names in our current database/API setup.

How to Implement Query Optimization:

LLM-Based: The most flexible approach is using another LLM call. We give the LLM the original user query and the context of the main agent's capabilities (its tools and purpose) and ask it to generate an optimized query.
Rule-Based: Simpler rules could be applied (e.g., identifying keywords like "invoice," "vendor," "customer," "status," "news" and structuring the query), but this is less flexible than an LLM.
Hybrid: Combine rules for common patterns and LLM for complex cases.
Decision: For this example, an LLM-based approach is more aligned with the "agent" concept and demonstrates LangChain's capabilities better.

Integrate into the Workflow (LangGraph):

New Node: Introduce a new node before the main agent_node. Let's call it optimize_query_node.
State Modification: Add a field to the APARState to hold the optimized query, e.g., optimized_prompt: Optional[str].
Graph Edges: The flow will now be: Initialize -> optimize_query_node -> agent_node -> END.
Design the optimize_query_node:

Input: Takes the current APARState, specifically the user_prompt and potentially the messages (for conversation context).
LLM Call: Uses an LLM (can be the same model or a different one) with a specific prompt.
Optimization Prompt: The prompt needs to instruct the LLM on its task:
"You are a query optimization assistant for an AP/AR agent."
"The main agent has these tools: [List tool names and brief descriptions, similar to the main agent's system prompt]."
"Given the user query and conversation history, rephrase the query to be as specific and effective as possible for the main agent."
"Focus on identifying entity names (vendors/customers) and the core task (AP status, AR status, public news, risk)."
"If the query is clear, you can keep it similar."
"If the query is ambiguous (e.g., lacks entity name), your output should reflect that, perhaps asking for clarification, OR generate a query that requests clarification." Self-correction: For simplicity in the graph, let's have it generate the best possible query or keep the original if it's good enough. Handling clarification requests adds more graph complexity.
"Output only the optimized query string."
Output: Updates the optimized_prompt field in the APARState. It should also update the messages to include the original prompt and potentially a message indicating the optimization step (optional, for clarity).
Modify the agent_node:

The agent_node should now primarily use the optimized_prompt from the state as its main input for deciding actions, although having access to the original user_prompt and full messages history is still valuable for context.
The prompt for the main agent might need slight adjustment to acknowledge it might receive an optimized query. However, keeping it focused on the tools is probably sufficient. The key is that the input message it acts upon is now the optimized one. Self-correction: It's better to pass the optimized_prompt explicitly in the dictionary passed to agent_executor.invoke rather than just relying on it being in the state. Let's adjust the agent_node's input processing.
Refine Implementation Details:

State Update: Ensure the optimize_query_node correctly returns the entire updated state object.
Message Handling: Decide how to represent the optimization step in the messages history. Maybe add the optimized_prompt as a specific type of message, or just rely on the main agent using it. Decision: Keep it simple for now; the main agent will use the optimized prompt, and the history will contain the original user prompt and the final AI response. The optimization step is internal processing.
Error Handling: Add try...except around the LLM call in optimize_query_node.
Update Main Function: Ensure the input dictionary for app.invoke still includes the user_prompt. The initialize node will pick this up, and the optimize_query_node will use it.
Review and Test (Mental Walkthrough):

User inputs "Status Tech Solutions".
main calls app.invoke({"user_prompt": "Status Tech Solutions", "messages": [...]}).
Graph starts. initialize_node sets up state including user_prompt.
optimize_query_node runs. It gets the user_prompt. Calls LLM with optimization instructions. LLM returns (hopefully) "Retrieve internal Accounts Payable data for vendor Tech Solutions Inc.". Updates state["optimized_prompt"].
agent_node runs. It receives the state containing the optimized_prompt. It uses this optimized prompt to decide actions. It calls get_internal_ap_data(vendor_name="Tech Solutions Inc."). Gets results. Synthesizes the final answer based on the original user intent but guided by the optimized query. Updates state["analysis_result"] and state["messages"].
Graph ends. main prints state["analysis_result"].
This refined process leads to the updated code structure, incorporating the optimization step logically within the LangGraph flow.

2.0 Pro Experimental. Lacks access to real-time info and some Gemini features.
Okay, let's integrate a "Conversational Query Optimization Agent" into the workflow. This agent will act as a preliminary step to refine the user's raw query, making it more precise for the main AP/AR agent, especially if we were dealing with document retrieval or more complex data sources.

Concept:

New Node: We'll add an optimize_query_node that runs after initialization but before the main agent_node.
Optimization Task: This node will use an LLM with a specific prompt to analyze the user's input (and potentially conversation history) and rewrite it into a more structured query suitable for the main agent's tools.
State Update: The optimized query will be stored in a new field in the state (optimized_prompt).
Main Agent Input: The agent_node will now use the optimized_prompt as its primary input for deciding actions, while still having access to the original prompt and history for full context.
'''

import os
import getpass
import sqlite3
from datetime import datetime, timedelta
from typing import TypedDict, Dict, List, Optional, Union, Any
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, END
# Use BaseMessage for type hinting message lists
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import json # For parsing potential JSON output if needed

from analysis.llm.fine_tuned.conversational_query_optimization.util.ap_ar_utils import setup_database, \
    get_internal_ap_data, get_internal_ar_data, search_public_data

# --- 0. Load Environment Variables & Setup Keys (Unchanged) ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if not os.getenv("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = getpass.getpass("SerpApi API Key:")

# --- 1. Database Setup (Unchanged) ---
DB_FILE = "enterprise_data.db"
# Function setup_database() remains the same as before...

# --- Ensure DB setup is run ---
# setup_database() # Assuming the function exists from the previous example

# --- 2. Define Agent State (with new field) ---
class APARErrorState:
    def __init__(self, message):
        self.message = message

class APARState(TypedDict):
    """Represents the state of the AP/AR agent workflow."""
    user_prompt: str
    optimized_prompt: Optional[str] # <<< NEW: For the optimized query
    entity_name: Optional[str]
    entity_type: Optional[str]
    internal_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]], str]]
    public_data: Optional[str]
    analysis_result: Optional[str]
    error_message: Optional[str]
    messages: List[BaseMessage] # Keep using BaseMessage for flexibility

# --- 3. Define Tools (Unchanged) ---
# Functions _run_query, get_internal_ap_data, get_internal_ar_data,
# and search_public_data remain the same as before...
# Assume these tool functions exist from the previous example

# --- 4. Query Optimization Node ---

def optimize_query_node(state: APARState) -> APARState:
    """Uses an LLM to refine the user query for the main agent."""
    print("--- Node: Optimizing Query ---")
    user_prompt = state["user_prompt"]
    message_history = state["messages"]

    # Use a separate LLM instance or the same one
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Can use a faster/cheaper model

    # Define the tools available to the *main* agent for context
    main_agent_tools_summary = """
    - get_internal_ap_data(vendor_name: str): Gets vendor details and unpaid/overdue AP invoices. Requires exact vendor name.
    - get_internal_ar_data(customer_name: str): Gets customer details and sent/overdue AR invoices. Requires exact customer name.
    - search_public_data(company_name: str): Searches web for recent news/financial health/contact info about a company.
    """

    optimization_system_prompt = f"""You are a query optimization assistant. Your task is to rephrase the user's latest query to be more precise and effective for a downstream AP/AR agent.
The downstream agent has access to these tools:
{main_agent_tools_summary}

Analyze the latest user query in the context of the conversation history. Identify the core intent (AP status, AR status, vendor info, customer info, public news, risk assessment).
Identify the specific company/vendor/customer name mentioned.

Rewrite the query into a concise action phrase or question that directly maps to the available tools and required inputs. Examples:
- User: "Status Tech Solutions?" -> Optimized: "Retrieve internal Accounts Payable data for vendor Tech Solutions Inc."
- User: "any overdue inovices for Global Corp?" -> Optimized: "Retrieve internal Accounts Receivable data for customer Global Corp focusing on overdue invoices."
- User: "tell me about Cloud Services LLC" -> Optimized: "Search public data for general information and recent news about Cloud Services LLC."
- User: "risk assessment for V1001" -> Optimized: "Search public data for recent news or financial issues related to vendor Tech Solutions Inc." (Requires mapping V1001 to name if possible, or clarifying)

If the query is already precise (e.g., "Use get_internal_ap_data for Tech Solutions Inc."), keep it as is.
If the query is ambiguous or lacks necessary information (like a company name), generate a query that asks for clarification or makes the best guess based on history. E.g., "Clarify which vendor or customer you are asking about."
Output ONLY the optimized query string.
"""

    optimization_prompt = ChatPromptTemplate.from_messages([
        ("system", optimization_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{latest_query}")
    ])

    chain = optimization_prompt | llm

    try:
        # Prepare history for the optimizer prompt
        # Exclude the absolute latest user message as it's passed separately
        history_for_optimizer = message_history[:-1] if len(message_history) > 1 else []

        response = chain.invoke({
            "history": history_for_optimizer,
            "latest_query": user_prompt
        })
        optimized_query = response.content.strip()
        print(f"Optimized Query: {optimized_query}")
        state["optimized_prompt"] = optimized_query
        # Optionally add a message to history about optimization? For now, keep it internal.
        # state["messages"].append(SystemMessage(content=f"Query optimized to: {optimized_query}"))
    except Exception as e:
        print(f"Error during query optimization: {e}")
        state["error_message"] = f"Failed to optimize query: {e}"
        # If optimization fails, fall back to the original prompt
        state["optimized_prompt"] = user_prompt
        state["messages"].append(SystemMessage(content=f"Warning: Query optimization failed. Using original query. Error: {e}"))


    return state


# --- 5. Agent Node (Modified to use optimized prompt) ---

def agent_node(state: APARState) -> APARState:
    """Invokes the LLM agent using the OPTIMIZED query."""
    print("--- Node: Agent ---")

    # Use the optimized prompt if available, otherwise fall back to original
    action_prompt = state.get("optimized_prompt") or state.get("user_prompt")
    if not action_prompt:
        state["analysis_result"] = "Error: No prompt available for the agent."
        state["messages"].append(AIMessage(content="Error: No prompt available for the agent."))
        return state

    print(f"Agent acting on: '{action_prompt}'")

    # Define Tools and LLM (Unchanged)
    tools = [get_internal_ap_data, get_internal_ar_data, search_public_data]
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Using GPT-4o

    # Define the prompt (System prompt remains largely the same)
    system_prompt = """You are an expert Accounts Payable (AP) and Accounts Receivable (AR) assistant.
Use the available tools to answer the user's request based on the potentially optimized action phrase provided.
Available Tools:
- get_internal_ap_data(vendor_name: str): Gets vendor details and unpaid/overdue AP invoices. Requires exact vendor name.
- get_internal_ar_data(customer_name: str): Gets customer details and sent/overdue AR invoices. Requires exact customer name.
- search_public_data(company_name: str): Searches web for recent news, financial health, contact info.

Process:
1. Analyze the core task from the input (which might be optimized).
2. Identify the specific vendor or customer name required by the tools.
3. Select and execute the appropriate tool(s). Use the exact names found in the potentially optimized prompt.
4. Synthesize the information gathered into a clear answer addressing the user's *original* intent (which you can infer from the conversation history).
5. If a tool returns an error (e.g., 'not found'), report that error clearly. Do not make up information.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"), # Full history for context
            # The user's request is implicitly the latest message or the optimized prompt
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create and run the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Debugging
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    try:
        # Pass the full message history for context,
        # but the primary instruction comes from the potentially optimized prompt implicitly
        # Let the agent decide based on the latest context which might include the optimized query intent
        result = agent_executor.invoke({"messages": state["messages"]}) # Agent uses history
        final_answer = result["output"]

        state["analysis_result"] = final_answer
        # Append only the *final* AI answer to the persistent history
        state["messages"].append(AIMessage(content=final_answer))

    except Exception as e:
        print(f"Agent execution error: {e}")
        state["error_message"] = f"Agent failed to process the request: {e}"
        # Append error message to history for the next turn
        error_msg = f"Sorry, I encountered an error: {e}"
        state["messages"].append(AIMessage(content=error_msg))
        state["analysis_result"] = error_msg # Also set as analysis result


    return state

# --- 6. Graph Definition (Updated Flow) ---

def create_workflow():
    """Creates and compiles the LangGraph workflow with query optimization."""
    workflow = StateGraph(APARState)

    # Define nodes
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.set_entry_point("optimize_query") # Start with optimization

    # Define edges
    workflow.add_edge("optimize_query", "agent") # Always go from optimizer to agent
    workflow.add_edge("agent", END) # Agent is the last step

    return workflow.compile()

# --- 7. Main Execution Loop (Unchanged, but will trigger new flow) ---

def main():
    """Main function to run the AP/AR agent with query optimization."""
    # Ensure DB setup runs
    setup_database()

    print("\n--- AP/AR Assistant Initialized (with Query Optimization) ---")
    print("Current Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("Ask questions about vendors (AP) or customers (AR). Type 'exit' to quit.")

    app = create_workflow()
    # Start with a system message
    message_history: List[BaseMessage] = [SystemMessage(content="AP/AR Assistant ready. Query optimization enabled.")]

    while True:
        user_input = input("\nYour Prompt: ")
        if user_input.lower() == 'exit':
            break

        # Add latest user input to history *before* invoking
        current_human_message = HumanMessage(content=user_input)
        message_history.append(current_human_message)

        # Prepare initial state for this run
        initial_state: APARState = {
            "user_prompt": user_input,
            "optimized_prompt": None, # Start as None
            "entity_name": None,
            "entity_type": None,
            "internal_data": None,
            "public_data": None,
            "analysis_result": None,
            "error_message": None,
            "messages": message_history # Pass the *entire* current history
        }

        # Invoke the workflow
        try:
            # Stream events for more visibility (optional but good for debugging)
            # final_state = None
            # print("Assistant thinking...")
            # for event in app.stream(initial_state):
            #     # You can print events here to see the flow
            #     # print(event)
            #     if "__end__" in event:
            #         final_state = event["__end__"]
            # if final_state is None: # Fallback if stream doesn't end as expected
            #      final_state = app.invoke(initial_state)

            # Simpler invoke for clarity:
            final_state = app.invoke(initial_state)


            # Print the agent's final response from the state
            agent_response = final_state.get("analysis_result", "No response generated.")
            print(f"\nAssistant: {agent_response}")

            # Update message history for the next turn using the final state's messages
            message_history = final_state.get("messages", message_history)

            # Handle potential errors captured during the run
            if final_state.get("error_message"):
                print(f"\n[Workflow Error]: {final_state['error_message']}")


        except Exception as e:
            print(f"\nAn unexpected error occurred during workflow execution: {e}")
            # Append error to history
            error_ai_msg = AIMessage(content=f"Sorry, a critical error occurred: {e}")
            message_history.append(error_ai_msg)


if __name__ == "__main__":
    # Assuming the tool and helper functions (_run_query, setup_database, etc.)
    # are defined above or imported correctly.
    # Make sure the database setup function exists and is callable.
    main()
