import json
import os
import getpass
import sqlite3
from datetime import datetime, timedelta
from typing import TypedDict, Dict, List, Optional, Union, Any
import pandas as pd

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document # Import Document schema
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# LangGraph Imports
from langgraph.graph import StateGraph, END

# Environment Loading
from dotenv import load_dotenv

from analysis.llm.fine_tuned.conversational_query_optimization.ap_ar import agent_node

# --- 0. Load Environment Variables & Setup Keys ---
load_dotenv()
# (API Key checks remain the same)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if not os.getenv("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = getpass.getpass("SerpApi API Key:")


# --- 1. Database Setup (Unchanged) ---
DB_FILE = "enterprise_data.db"
# setup_database() function remains the same...
# Ensure setup_database() is called if needed before running main()

# --- 1b. Simulated Internal Documents & Vector Store Setup ---

# Sample Documents (Replace with actual loading from files/DB in practice)
SAMPLE_DOCS = [
    Document(
        page_content="Vendor Agreement: Tech Solutions Inc. (V1001)\nEffective Date: 2023-01-15\nTerm: 2 years\nPayment Terms: NET 30\nService Level Agreement: Standard Tier Support includes response within 4 business hours for critical issues. Contact: ap@techsolutions.com.\nRenewal Clause: Auto-renews unless notified 60 days prior.",
        metadata={"source": "contracts", "entity_id": "V1001", "doc_type": "vendor_agreement"}
    ),
    Document(
        page_content="Accounts Payable Policy\nStandard payment terms are NET 30 unless specified otherwise in vendor contract. Early payment discounts should be taken when advantageous (>= 2%). Invoices over $10,000 require VP approval. Overdue invoices trigger automated reminders after 5 days past due. Contact AP department for exceptions.",
        metadata={"source": "policies", "doc_type": "ap_policy"}
    ),
    Document(
        page_content="Customer Onboarding Notes: Global Corp (C2001)\nOnboarding Date: 2022-11-01\nPrimary Contact: finance@globalcorp.com\nAgreed Credit Limit: $10,000. Payment History: Generally reliable, occasional delays up to 5 days past due on larger invoices. Discussed NET 30 terms. No special discount agreements.",
        metadata={"source": "crm_notes", "entity_id": "C2001", "doc_type": "customer_notes"}
    ),
    Document(
        page_content="Vendor Performance Review Notes: Office Supplies Co. (V1002)\nDate: 2024-03-10\nSummary: Reliable delivery, occasional minor backorders on specialty items. Pricing is competitive. Payment terms NET 15 strictly enforced by vendor. No major issues reported. Continue relationship.",
        metadata={"source": "procurement_notes", "entity_id": "V1002", "doc_type": "vendor_review"}
    ),
]

# Global variable for the retriever (simpler for demo)
# In production, manage this more robustly (e.g., pass via state or context)
vector_store_retriever = None

def setup_vector_store():
    """Creates an in-memory FAISS vector store from sample documents."""
    global vector_store_retriever
    if vector_store_retriever:
        print("Vector store already set up.")
        return

    print("Setting up vector store...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(SAMPLE_DOCS)

        embeddings = OpenAIEmbeddings() # Uses OPENAI_API_KEY

        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store_retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
        print("Vector store setup complete.")

    except Exception as e:
        print(f"Error setting up vector store: {e}")
        print("Document retrieval features will be unavailable.")
        vector_store_retriever = None # Ensure it's None if setup fails

# --- Run vector store setup ---
# setup_vector_store() # Call this once before main loop

# --- 2. Define Agent State (with new fields) ---

class APARState(TypedDict):
    """Represents the state of the AP/AR agent workflow."""
    user_prompt: str
    optimized_prompt: Optional[str]
    retrieved_docs_content: Optional[str] # <<< NEW: Content of retrieved docs
    direct_answer_found: bool # <<< NEW: Flag if optimizer answered directly
    entity_name: Optional[str]
    entity_type: Optional[str]
    internal_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]], str]]
    public_data: Optional[str]
    analysis_result: Optional[str]
    error_message: Optional[str]
    messages: List[BaseMessage]

# --- 3. Define Tools (Unchanged) ---
# _run_query, get_internal_ap_data, get_internal_ar_data, search_public_data
# Assume these functions exist from previous examples...

# --- 4. Query Optimization Node (Enhanced with Document Retrieval) ---

def optimize_query_node(state: APARState) -> APARState:
    """Uses LLM and retrieved documents to refine the query or answer directly."""
    print("--- Node: Optimizing Query (with Document Retrieval) ---")
    user_prompt = state["user_prompt"]
    message_history = state["messages"]
    state["direct_answer_found"] = False # Reset flag

    # --- Document Retrieval ---
    retrieved_docs_content = "No relevant documents found."
    if vector_store_retriever:
        try:
            retrieved_docs = vector_store_retriever.invoke(user_prompt)
            if retrieved_docs:
                retrieved_docs_content = "\n\n".join(
                    [f"--- Document Snippet (Source: {doc.metadata.get('source', 'N/A')}) ---\n{doc.page_content}" for doc in retrieved_docs]
                )
                print(f"Retrieved {len(retrieved_docs)} document snippets.")
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            retrieved_docs_content = f"Error retrieving documents: {e}"
    else:
        print("Skipping document retrieval (vector store not available).")

    state["retrieved_docs_content"] = retrieved_docs_content # Store for potential use later if needed

    # --- LLM Optimization ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Use a capable model

    main_agent_tools_summary = """
    - get_internal_ap_data(vendor_name: str): Gets vendor details and unpaid/overdue AP invoices. Requires exact vendor name.
    - get_internal_ar_data(customer_name: str): Gets customer details and sent/overdue AR invoices. Requires exact customer name.
    - search_public_data(company_name: str): Searches web for recent news/financial health/contact info.
    """

    optimization_system_prompt = f"""You are a query analysis and optimization assistant for an AP/AR agent.
Your goal is to either directly answer the user's query using the provided document context OR refine the query for a downstream agent.

Context Provided:
1. User's latest query.
2. Conversation History.
3. Relevant snippets from internal documents (if found).

Downstream Agent Tools (if needed):
{main_agent_tools_summary}

Analysis Steps:
1. Understand the user's latest query in light of the conversation history.
2. Examine the retrieved document snippets. Does the context *directly and completely* answer the user's query?
3. **If YES (documents answer the query):** Synthesize a concise answer *based ONLY on the document snippets*. Start your answer with "[Answer from Documents]". Set 'direct_answer_found' to true in the output JSON.
4. **If NO (documents don't fully answer or are irrelevant):** Rewrite the user's query into a precise action phrase or question for the downstream agent. Focus on extracting entity names and the core task (AP status, AR status, public news, etc.) needed for the downstream tools. Set 'direct_answer_found' to false.
5. **If the query is ambiguous:** Generate an optimized query asking for clarification. Set 'direct_answer_found' to false.

Output Format: Return a JSON object with two keys:
- "optimized_query_or_answer": (string) Your synthesized answer (if direct_answer_found is true) OR the optimized query string (if direct_answer_found is false).
- "direct_answer_found": (boolean) True if you generated a direct answer from documents, False otherwise.

Example Output 1 (Direct Answer):
{{
  "optimized_query_or_answer": "[Answer from Documents] According to the vendor agreement, Tech Solutions Inc.'s payment terms are NET 30.",
  "direct_answer_found": true
}}
Example Output 2 (Optimized Query):
{{
  "optimized_query_or_answer": "Retrieve internal Accounts Payable data for vendor Tech Solutions Inc., focusing on overdue invoices.",
  "direct_answer_found": false
}}
Example Output 3 (Clarification Query):
{{
  "optimized_query_or_answer": "Which vendor or customer are you asking about payment status for?",
  "direct_answer_found": false
}}
"""

    optimization_prompt = ChatPromptTemplate.from_messages([
        ("system", optimization_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "User Query: {latest_query}\n\nRetrieved Document Context:\n{document_context}")
    ])

    chain = optimization_prompt | llm

    try:
        history_for_optimizer = message_history[:-1] if len(message_history) > 1 else []
        response = chain.invoke({
            "history": history_for_optimizer,
            "latest_query": user_prompt,
            "document_context": retrieved_docs_content
        })
        response_content = response.content.strip()

        # Attempt to parse JSON output
        try:
            parsed_output = json.loads(response_content)
            optimized_query = parsed_output.get("optimized_query_or_answer", user_prompt) # Fallback
            direct_answer_found = parsed_output.get("direct_answer_found", False)
            print(f"Optimizer Output: {optimized_query}")
            print(f"Direct Answer Found: {direct_answer_found}")

            state["optimized_prompt"] = optimized_query # Store query OR direct answer
            state["direct_answer_found"] = direct_answer_found

            # If it's a direct answer, store it as the final analysis result immediately
            if direct_answer_found:
                state["analysis_result"] = optimized_query
                state["messages"].append(AIMessage(content=optimized_query)) # Add direct answer to history

        except json.JSONDecodeError:
            print(f"Warning: Optimizer did not return valid JSON. Output: {response_content}")
            # Fallback: Treat the whole output as an optimized query
            state["optimized_prompt"] = response_content
            state["direct_answer_found"] = False # Assume no direct answer if JSON fails

    except Exception as e:
        print(f"Error during query optimization LLM call: {e}")
        state["error_message"] = f"Failed to optimize query: {e}"
        state["optimized_prompt"] = user_prompt # Fallback
        state["direct_answer_found"] = False
        state["messages"].append(SystemMessage(content=f"Warning: Query optimization failed. Using original query. Error: {e}"))

    return state


# --- 5. Agent Node (Largely Unchanged, but now uses potentially optimized prompt's intent) ---
# agent_node function remains the same as the previous *working* version...
# It will implicitly use the state["optimized_prompt"]'s intent via the message history.

# --- 6. Conditional Logic ---

def check_for_direct_answer(state: APARState) -> str:
    """Determines the next step based on whether a direct answer was found."""
    print(f"--- Node: Checking for Direct Answer (Found: {state.get('direct_answer_found', False)}) ---")
    if state.get("direct_answer_found", False):
        return "end_workflow" # Go directly to END if optimizer answered
    else:
        # Make sure there's a prompt for the agent
        action_prompt = state.get("optimized_prompt") or state.get("user_prompt")
        if not action_prompt:
            print("Error: No prompt available for the agent after optimization.")
            # Handle this edge case - maybe go to end or raise an error state
            return "end_workflow" # Or another error handling node
        return "continue_to_agent" # Proceed to the main agent

# --- 7. Graph Definition (Updated with Conditional Edge) ---

def create_workflow():
    """Creates and compiles the LangGraph workflow with optimization and conditional routing."""
    workflow = StateGraph(APARState)

    # Define nodes
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("agent", agent_node) # Main agent remains

    # Set entry point
    workflow.set_entry_point("optimize_query")

    # Add conditional edge from optimizer
    workflow.add_conditional_edges(
        "optimize_query",
        check_for_direct_answer,
        {
            "continue_to_agent": "agent",
            "end_workflow": END
        }
    )

    # Add edge from agent to end
    workflow.add_edge("agent", END)

    return workflow.compile()

# --- 8. Main Execution Loop ---

def main():
    """Main function to run the AP/AR agent with query optimization and document retrieval."""
    # --- Run Setup Functions ---
    # setup_database() # Make sure DB exists
    setup_vector_store() # Setup vector store at the start

    print("\n--- AP/AR Assistant Initialized (with Doc Retrieval & Query Optimization) ---")
    print("Current Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("Ask questions about vendors (AP) or customers (AR). Type 'exit' to quit.")

    app = create_workflow()
    message_history: List[BaseMessage] = [SystemMessage(content="AP/AR Assistant ready. Query optimization and document retrieval enabled.")]

    while True:
        user_input = input("\nYour Prompt: ")
        if user_input.lower() == 'exit':
            break

        current_human_message = HumanMessage(content=user_input)
        message_history.append(current_human_message)

        initial_state: APARState = {
            "user_prompt": user_input,
            "optimized_prompt": None,
            "retrieved_docs_content": None,
            "direct_answer_found": False, # Initialize flag
            "entity_name": None,
            "entity_type": None,
            "internal_data": None,
            "public_data": None,
            "analysis_result": None,
            "error_message": None,
            "messages": message_history # Pass full history
        }

        try:
            final_state = app.invoke(initial_state)

            # The final answer is now either analysis_result (set by optimizer or agent)
            agent_response = final_state.get("analysis_result", "No response generated.")
            print(f"\nAssistant: {agent_response}")

            message_history = final_state.get("messages", message_history)
            if final_state.get("error_message"):
                print(f"\n[Workflow Error]: {final_state['error_message']}")

        except Exception as e:
            print(f"\nAn unexpected error occurred during workflow execution: {e}")
            error_ai_msg = AIMessage(content=f"Sorry, a critical error occurred: {e}")
            message_history.append(error_ai_msg)

if __name__ == "__main__":
    # Make sure all required functions (setup_database, tools, agent_node)
    # are defined above in the actual script.
    main()