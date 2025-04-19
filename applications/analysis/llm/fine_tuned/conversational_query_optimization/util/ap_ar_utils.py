'''
Here's a breakdown of the thought process behind constructing the AP/AR LLM agent example:

Understand the Core Request: The user wants an end-to-end Python implementation of an LLM-based agent focused on Accounts Payable (AP) for vendors and Accounts Receivable (AR) for an enterprise. This agent needs to use both internal data (likely from ERP/accounting systems) and public data (like news, credit reports, market trends).

Identify Key Use Cases: What would such an agent do?

AP (Vendors):
Check vendor payment status.
Assess vendor risk (late payments, financial distress based on news).
Get vendor contact information.
Analyze spending patterns with a vendor.
Suggest optimal payment timing (considering discounts, cash flow).
Identify potential duplicate invoices/payments (though this often requires specialized logic beyond a simple LLM).
AR (Customers):
Check customer invoice status/outstanding balance.
Assess customer creditworthiness/payment risk (using internal history and public data).
Get customer contact information.
Analyze customer payment trends.
Predict likelihood of late payment.
Suggest follow-up actions for overdue invoices.
Determine Necessary Data Sources:

Internal:
ERP/Accounting System Database (Vendor Master, Customer Master, AP Invoices, AR Invoices, Payment History, GL). This is the most critical component.
Potentially CRM data for customer contacts/interactions.
Public:
News APIs (e.g., Google News, NewsAPI).
Financial Data APIs (e.g., Yahoo Finance, Alpha Vantage for public companies).
Credit Rating Agencies (Requires subscriptions/APIs - harder to simulate).
Web Search (General company information, potential distress signals).
Choose the Right Framework: LangGraph is explicitly requested and suitable for managing the multi-step process of data retrieval, analysis, and response generation. LangChain provides the core agent framework, LLM integration, and tool definition capabilities.

Design the Agent Architecture:

State: What information needs to be tracked throughout the agent's process?
User query/prompt.
Identified entity (vendor name, customer name, invoice number).
Internal data retrieved (vendor details, invoice status, payment history).
Public data retrieved (news snippets, basic financial info).
Analysis results (risk assessment, payment prediction).
Final agent response.
Conversation history (messages).
Tools: What actions can the agent perform? Each data source/analysis type needs a tool.
get_internal_vendor_data: Query the internal DB for vendor info, invoices, payments.
get_internal_customer_data: Query the internal DB for customer info, invoices, payments.
search_web_for_company_info: Use a search engine (like Google Search via an API) for general news and information. Self-correction: Initially thought about specific news APIs, but a general web search is more versatile for a demo.
analyze_payment_risk (Optional but good): A tool that potentially combines internal/public data to give a risk score. This could be a simple rule-based system or even call another LLM with specific instructions. Self-correction: Keep this simple for the demo.
Agent Logic (LLM Prompt): How should the agent behave?
Identify the core task (AP or AR).
Identify the specific entity (vendor/customer).
Determine which tools are needed based on the query.
Execute tools sequentially or in parallel where appropriate.
Synthesize the results from different tools.
Formulate a comprehensive answer.
Handle ambiguity or missing data gracefully.
Graph Structure: A sequential flow seems appropriate for most queries: Initialize -> Identify Entity (Maybe) -> Fetch Internal Data -> Fetch Public Data -> Analyze (Optional) -> Generate Response. LangGraph makes this easy to define.
Flesh out the Implementation Details:

Database Simulation: Since connecting to a real ERP is complex and environment-specific, simulate it using SQLite. Create realistic-looking tables (Vendors, Customers, AP_Invoices, AR_Invoices). Populate with sample data.
Public Data Simulation/API: Use the google-search-results library (or similar) for web search. Set up API keys (mentioning the need for real keys). Self-correction: Emphasize using environment variables for API keys.
Tool Implementation:
Database tools will execute SQL queries against the SQLite DB using sqlite3. Use parameterized queries (?) for security. Handle cases where data isn't found.
Web search tool will call the search API and return formatted snippets.
Agent Node (agent_node):
Initialize the LLM (e.g., ChatOpenAI).
Define the system prompt carefully, explaining the agent's role, capabilities, and available tools.
Use create_openai_functions_agent.
Create the AgentExecutor. Crucially, ensure the state (or relevant parts of it) is accessible to the tools (using the RunnableLambda approach from the previous correction is best).
Manage the messages for conversation history.
Graph (create_workflow): Define nodes and edges. Keep it relatively simple for the demo.
Main Function (main): Create the interactive loop to get user input and invoke the graph. Pass the user_prompt in the inputs.
Refine and Add Best Practices:

Error Handling: Add try...except blocks, especially around database connections and API calls.
Security: Emphasize secure handling of credentials (environment variables). Warn against hardcoding. Mention parameterized SQL queries.
Clarity and Comments: Add comments explaining complex parts. Use clear variable names. Use type hints.
Limitations: Explicitly state the limitations of the demo (simulated data, basic analysis, need for real API keys, potential cost).
Scalability: Briefly mention how this could be scaled (real DB connection, more sophisticated tools, asynchronous processing).
Date/Time: Add the current date/time context as requested by the user prompt meta-information.
Review and Test (Mental Walkthrough): Reread the code. Does the flow make sense? Does the agent have the information it needs at each step? Does the tool access the state correctly? Does the main function pass the input correctly? (This is where the previous errors were caught). Ensure all necessary imports are present. Check for typos.

This structured thinking process, moving from high-level requirements to specific implementation details while incorporating best practices and anticipating potential issues, helps create a comprehensive and functional example like the one provided.

Okay, let's build an end-to-end Python implementation for an LLM-based agent focusing on Accounts Payable (AP) and Accounts Receivable (AR). This agent will use simulated internal data (SQLite database) and real public data (via web search).

Core Concepts:

Internal Data Simulation: We'll use SQLite to mimic an ERP/accounting database containing vendor, customer, AP invoice, and AR invoice data.
Public Data Integration: We'll use a web search tool (requiring an API key like SerpApi or similar) to fetch recent news or general information about vendors/customers.
LangChain Agent: We'll use LangChain's agent framework (create_openai_functions_agent and AgentExecutor) to allow the LLM to reason and decide which tools to use.
LangGraph Workflow: We'll structure the process using LangGraph to manage the state and the flow between data retrieval, analysis (by the agent), and response generation.
Tools: We'll define specific tools for the agent to:
Query internal AP data (vendor info, invoice status).
Query internal AR data (customer info, invoice status).
Search the web for public information about a company.
'''


import os
import getpass
import sqlite3
from datetime import datetime, timedelta
from typing import TypedDict, Dict, List, Optional, Any, Union
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_community.utilities import SerpAPIWrapper # Using SerpApi for search
from dotenv import load_dotenv

# --- 0. Load Environment Variables ---
load_dotenv()

# --- Check for API Keys (Improved Handling) ---
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if not os.getenv("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = getpass.getpass("SerpApi API Key:")


# --- 1. Database Setup (Simulated Internal Data) ---

DB_FILE = "enterprise_data.db"

def setup_database():
    """Creates and populates the SQLite database if it doesn't exist."""
    if os.path.exists(DB_FILE):
        print("Database already exists.")
        return

    print("Setting up simulated database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create Tables
    cursor.execute("""
        CREATE TABLE vendors (
            vendor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            contact_email TEXT,
            payment_terms TEXT DEFAULT 'NET 30'
        )
    """)
    cursor.execute("""
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            credit_limit REAL DEFAULT 5000.00,
            contact_email TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE ap_invoices (
            invoice_id TEXT PRIMARY KEY,
            vendor_id TEXT,
            invoice_date DATE,
            due_date DATE,
            amount REAL,
            status TEXT DEFAULT 'Unpaid', -- Unpaid, Paid, Overdue
            payment_date DATE,
            FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE ar_invoices (
            invoice_id TEXT PRIMARY KEY,
            customer_id TEXT,
            invoice_date DATE,
            due_date DATE,
            amount REAL,
            status TEXT DEFAULT 'Sent', -- Sent, Paid, Overdue
            payment_date DATE,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
    """)

    # Populate Sample Data
    vendors_data = [
        ('V1001', 'Tech Solutions Inc.', 'ap@techsolutions.com', 'NET 30'),
        ('V1002', 'Office Supplies Co.', 'billing@officesupplies.com', 'NET 15'),
        ('V1003', 'Cloud Services LLC', 'accounts@cloudsvc.com', 'NET 45')
    ]
    customers_data = [
        ('C2001', 'Global Corp', 10000.00, 'finance@globalcorp.com'),
        ('C2002', 'Innovate Ltd.', 7500.00, 'accounts@innovate.com'),
        ('C2003', 'Startup Ventures', 2500.00, 'billing@startupvent.com')
    ]
    today = datetime.now().date()
    ap_invoices_data = [
        ('AP-001', 'V1001', today - timedelta(days=20), today + timedelta(days=10), 1500.00, 'Unpaid', None),
        ('AP-002', 'V1002', today - timedelta(days=10), today + timedelta(days=5), 250.50, 'Unpaid', None),
        ('AP-003', 'V1001', today - timedelta(days=40), today - timedelta(days=10), 1200.00, 'Paid', today - timedelta(days=9)),
        ('AP-004', 'V1003', today - timedelta(days=60), today - timedelta(days=15), 5000.00, 'Overdue', None),
    ]
    ar_invoices_data = [
        ('AR-101', 'C2001', today - timedelta(days=15), today + timedelta(days=15), 3000.00, 'Sent', None),
        ('AR-102', 'C2002', today - timedelta(days=5), today + timedelta(days=25), 4500.75, 'Sent', None),
        ('AR-103', 'C2001', today - timedelta(days=50), today - timedelta(days=20), 2500.00, 'Paid', today - timedelta(days=18)),
        ('AR-104', 'C2003', today - timedelta(days=45), today - timedelta(days=15), 1000.00, 'Overdue', None),
    ]

    cursor.executemany("INSERT INTO vendors VALUES (?, ?, ?, ?)", vendors_data)
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers_data)
    cursor.executemany("INSERT INTO ap_invoices VALUES (?, ?, ?, ?, ?, ?, ?)", ap_invoices_data)
    cursor.executemany("INSERT INTO ar_invoices VALUES (?, ?, ?, ?, ?, ?, ?)", ar_invoices_data)

    conn.commit()
    conn.close()
    print("Database setup complete.")

# --- Run setup ---
setup_database()

# --- 2. Define Agent State ---

class APARErrorState: #Simple error state
    def __init__(self, message):
        self.message = message
class APARState(TypedDict):
    """Represents the state of the AP/AR agent workflow."""
    user_prompt: str
    entity_name: Optional[str] # Vendor or Customer name
    entity_type: Optional[str] # 'vendor' or 'customer'
    internal_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]], str]] # Dataframes or lists of dicts or error messages
    public_data: Optional[str] # Search results or error message
    analysis_result: Optional[str] # Agent's synthesis
    error_message: Optional[str] # For capturing errors during execution
    messages: List[BaseMessage] # Conversation history

# --- 3. Define Tools ---

def _run_query(query: str, params: tuple = ()) -> Union[List[Dict[str, Any]], APARErrorState]:
    """Helper function to run a query and return results as list of dicts."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        return APARErrorState(f"Database error accessing internal data: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return APARErrorState(f"An unexpected error occurred: {e}")


@tool
def get_internal_ap_data(vendor_name: str) -> Union[str, APARErrorState]:
    """
    Retrieves internal Accounts Payable data for a specific vendor, including
    vendor details and their unpaid or overdue invoices.
    Args: vendor_name: The exact name of the vendor.
    """
    print(f"--- Tool: get_internal_ap_data (Vendor: {vendor_name}) ---")
    # Find vendor_id
    vendor_query = "SELECT vendor_id, contact_email, payment_terms FROM vendors WHERE name = ?"
    vendor_result = _run_query(vendor_query, (vendor_name,))
    if isinstance(vendor_result, APARErrorState): return vendor_result # Pass error state
    if not vendor_result:
        return f"Vendor '{vendor_name}' not found in internal records."

    vendor_info = vendor_result[0]
    vendor_id = vendor_info['vendor_id']

    # Find invoices
    invoice_query = """
        SELECT invoice_id, invoice_date, due_date, amount, status
        FROM ap_invoices
        WHERE vendor_id = ? AND status IN ('Unpaid', 'Overdue')
        ORDER BY due_date
    """
    invoice_result = _run_query(invoice_query, (vendor_id,))
    if isinstance(invoice_result, APARErrorState): return invoice_result

    if not invoice_result:
        return f"Vendor Info: {vendor_info}. No unpaid or overdue AP invoices found for {vendor_name}."

    # Format results nicely
    df = pd.DataFrame(invoice_result)
    return f"Vendor Info: {vendor_info}\nUnpaid/Overdue AP Invoices:\n{df.to_string()}"

@tool
def get_internal_ar_data(customer_name: str) -> Union[str, APARErrorState]:
    """
    Retrieves internal Accounts Receivable data for a specific customer, including
    customer details and their sent or overdue invoices.
    Args: customer_name: The exact name of the customer.
    """
    print(f"--- Tool: get_internal_ar_data (Customer: {customer_name}) ---")
    # Find customer_id
    customer_query = "SELECT customer_id, contact_email, credit_limit FROM customers WHERE name = ?"
    customer_result = _run_query(customer_query, (customer_name,))
    if isinstance(customer_result, APARErrorState): return customer_result # Pass error state
    if not customer_result:
        return f"Customer '{customer_name}' not found in internal records."

    customer_info = customer_result[0]
    customer_id = customer_info['customer_id']

    # Find invoices
    invoice_query = """
        SELECT invoice_id, invoice_date, due_date, amount, status
        FROM ar_invoices
        WHERE customer_id = ? AND status IN ('Sent', 'Overdue')
        ORDER BY due_date
    """
    invoice_result = _run_query(invoice_query, (customer_id,))
    if isinstance(invoice_result, APARErrorState): return invoice_result

    if not invoice_result:
        return f"Customer Info: {customer_info}. No sent or overdue AR invoices found for {customer_name}."

    # Format results nicely
    df = pd.DataFrame(invoice_result)
    return f"Customer Info: {customer_info}\nSent/Overdue AR Invoices:\n{df.to_string()}"

@tool
def search_public_data(company_name: str) -> str:
    """
    Searches the web (Google) for recent news or general information about a company.
    Useful for assessing risk or finding contact details not in internal systems.
    Args: company_name: The name of the company (vendor or customer) to search for.
    """
    print(f"--- Tool: search_public_data (Company: {company_name}) ---")
    try:
        # Ensure the API key is set for the SerpAPIWrapper
        if not os.getenv("SERPAPI_API_KEY"):
            return APARErrorState("SERPAPI_API_KEY environment variable not set.")

        search = SerpAPIWrapper()
        # Query for recent news or potential issues
        query = f"{company_name} recent news OR financial health OR payment issues"
        results = search.run(query)
        if not results or "No good search result found" in results:
            # Fallback to simpler query if no news found
            query = f"{company_name} official website contact information"
            results = search.run(query)

        # Limit the length to avoid overwhelming the LLM
        return f"Web Search Results for '{company_name}':\n{results[:1500]}" # Limit length

    except Exception as e:
        print(f"Error during web search for {company_name}: {e}")
        # Return an error state or message
        return APARErrorState(f"Error searching public data for {company_name}: {e}")


# --- 4. Agent Node ---

def agent_node(state: APARState) -> APARState:
    """Invokes the LLM agent to decide actions and synthesize information."""
    print("--- Node: Agent ---")

    # Define Tools and LLM
    tools = [get_internal_ap_data, get_internal_ar_data, search_public_data]
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Use a capable model like GPT-4

    # Define the prompt
    # Instruct the agent on its role, data sources, and how to use tools.
    system_prompt = """You are an expert Accounts Payable (AP) and Accounts Receivable (AR) assistant for an enterprise.
Your goal is to answer user questions accurately using available tools.
Available Tools:
- get_internal_ap_data: Use for questions about specific vendors, their payment terms, and UNPAID/OVERDUE AP invoices. Requires the exact vendor name.
- get_internal_ar_data: Use for questions about specific customers, their credit limits, and SENT/OVERDUE AR invoices. Requires the exact customer name.
- search_public_data: Use to find recent news, assess potential risks (financial distress, payment issues), or find general info about a vendor or customer if internal data is insufficient or if specifically asked.

Process:
1. Identify if the query is about AP (vendors) or AR (customers).
2. Identify the specific vendor or customer name mentioned. Be precise.
3. If the name is found, use the appropriate internal data tool (`get_internal_ap_data` or `get_internal_ar_data`).
4. If the user asks about risk, news, or general info, OR if internal data is insufficient, use `search_public_data`. You might use both internal and public tools for comprehensive answers (e.g., checking internal status AND public news for risk).
5. Synthesize the information gathered from the tools into a clear, concise answer to the user's original prompt.
6. If a tool returns an error message (like 'Vendor not found' or database errors), report that error clearly to the user. Do not try to make up information.
7. If the query is ambiguous or lacks a specific entity name, ask the user for clarification.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"), # Include conversation history
            # No explicit "user" message here, it's part of the history
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create and run the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True for debugging agent steps
        handle_parsing_errors=True,
        return_intermediate_steps=False, # Don't need these in the final state for this example
    )

    # Invoke the agent with current messages and state context if needed
    # For this structure, message history is usually sufficient
    try:
        result = agent_executor.invoke({"messages": state["messages"]})
        # Update state with agent's final response
        state["messages"].append(AIMessage(content=result["output"]))
        state["analysis_result"] = result["output"] # Store the final answer
    except Exception as e:
        print(f"Agent execution error: {e}")
        state["error_message"] = f"Agent failed to process the request: {e}"
        # Add an error message to the history
        state["messages"].append(AIMessage(content=f"Sorry, I encountered an error: {e}"))


    return state

# --- 5. Graph Definition ---

def create_workflow():
    """Creates and compiles the LangGraph workflow."""
    workflow = StateGraph(APARState)

    # Define nodes
    # We don't need separate nodes for each tool call IF the agent decides which to call.
    # The 'agent' node will handle the logic of calling tools.
    # We might add nodes later for more complex pre/post-processing if needed.
    workflow.add_node("agent", agent_node)

    # Set entry point
    # The agent node is effectively the entry point after initialization
    # (which happens implicitly when invoking with initial state)
    workflow.set_entry_point("agent")

    # Define edges
    # For this agent-centric flow, the agent node leads directly to the end.
    # If we had separate tool nodes, we'd have conditional edges based on agent output.
    workflow.add_edge("agent", END)

    return workflow.compile()

# --- 6. Main Execution Loop ---

def main():
    """Main function to run the AP/AR agent."""
    print("\n--- AP/AR Assistant Initialized ---")
    print("Current Date:", datetime.now().strftime("%Y-%m-%d"))
    print("Ask questions about vendors (AP) or customers (AR). Type 'exit' to quit.")

    app = create_workflow()
    # Initial message list - could be empty or start with a greeting
    message_history = [SystemMessage(content="AP/AR Assistant ready.")]

    while True:
        user_input = input("\nYour Prompt: ")
        if user_input.lower() == 'exit':
            break

        # Add user input to history
        message_history.append(HumanMessage(content=user_input))

        # Prepare initial state for this run
        # Pass the LATEST message history
        initial_state: APARState = {
            "user_prompt": user_input,
            "entity_name": None,
            "entity_type": None,
            "internal_data": None,
            "public_data": None,
            "analysis_result": None,
            "error_message": None,
            "messages": message_history # Pass current history
        }

        # Invoke the workflow
        try:
            final_state = app.invoke(initial_state)

            # Print the agent's final response from the state
            agent_response = final_state.get("analysis_result", "No response generated.")
            print(f"\nAssistant: {agent_response}")

            # Update message history for the next turn
            # The agent node should have already appended the AIMessage
            message_history = final_state.get("messages", message_history)

            # Handle potential errors captured during the run
            if final_state.get("error_message"):
                print(f"\n[Workflow Error]: {final_state['error_message']}")


        except Exception as e:
            print(f"\nAn unexpected error occurred during workflow execution: {e}")
            # Optionally, reset history or handle error more gracefully
            message_history.append(AIMessage(content=f"Sorry, a critical error occurred: {e}"))


if __name__ == "__main__":
    main()
