# assistant_logic.py
from __future__ import annotations

import re
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# Import dummy data from config (or real data access methods)
from config import DUMMY_INTERNAL_DATA, DUMMY_PUBLIC_DATA, DEFAULT_CHAT_MODEL, FALLBACK_CHAT_MODEL

# --- Tool Functions ---

def _get_internal_financial_data(query: str) -> dict | str:
    """
    !! SECURITY RISK !! DEMO ONLY !!
    Retrieves specific internal financial data points from DUMMY data.
    Input format should clearly state the metric, unit, and period.
    Example query: 'Get Current Assets for Unit Alpha for 2025Q1'.
    Returns a dictionary like {'Current Assets': 1200000} or an error string.
    """
    print(f"\n--- TOOL CALL: InternalFinancialDataLookup ---")
    print(f"Query: {query}")
    query_lower = query.lower()
    match = re.search(r"get (.*?) for (.*?) for (\d{4}q\d)", query_lower)
    if not match: match = re.search(r"what is the (.*?) for (.*?) in (\d{4}q\d)", query_lower)

    if match:
        metric_str, unit_str, period_str = [s.strip().title() for s in match.groups()]
        period_str = period_str.upper()
        try:
            value = DUMMY_INTERNAL_DATA[unit_str][period_str][metric_str]
            print(f"Found: {metric_str}={value} for {unit_str} in {period_str}")
            return {metric_str: value}
        except KeyError:
            error_msg = f"Data not found for Metric: '{metric_str}', Unit: '{unit_str}', Period: '{period_str}'."
            print(f"Error: {error_msg}")
            return error_msg
    else:
        error_msg = "Could not parse internal data query. Use format: 'Get [Metric] for [Unit] for [YYYYQQ]'."
        print(f"Error: {error_msg}")
        return error_msg

def _get_public_market_data(query: str) -> dict | str:
    """Retrieves public market information or industry benchmarks from DUMMY data."""
    print(f"\n--- TOOL CALL: PublicMarketDataLookup ---")
    print(f"Query: {query}")
    query_lower = query.lower()
    data = None
    if "tech industry benchmark" in query_lower: data = DUMMY_PUBLIC_DATA["tech industry benchmark"]
    elif "manufacturing industry benchmark" in query_lower: data = DUMMY_PUBLIC_DATA["manufacturing industry benchmark"]
    else: return "Public data not found for this query in the dummy store."

    for key in data:
        if key.lower() in query_lower: return {key: data[key]}
    return data # Return all benchmark data if specific key not found

def _calculate_liquidity_ratio(input_data: dict) -> str:
    """Calculates Current Ratio or Quick Ratio from input dictionary."""
    print(f"\n--- TOOL CALL: LiquidityRatioCalculator ---")
    print(f"Input Data: {input_data}")
    try:
        ratio_type = input_data['ratio_to_calculate']
        liabilities = float(input_data['Current Liabilities'])
        if liabilities == 0: return "Error: Cannot calculate ratio with zero Current Liabilities."
        assets = float(input_data['Current Assets'])

        if ratio_type == 'Current Ratio': result = assets / liabilities
        elif ratio_type == 'Quick Ratio':
            inventory = float(input_data['Inventory'])
            result = (assets - inventory) / liabilities
        else: return f"Error: Calculation for '{ratio_type}' not supported."
        return f"Calculated {ratio_type}: {result:.2f}"
    except KeyError as e: return f"Error: Missing required value for calculation: {e}"
    except ValueError: return "Error: Input values must be numeric."
    except Exception as e: return f"Error during calculation: {e}"

# --- Tool Creation ---

def create_tools() -> list[Tool]:
    """Creates and returns a list of Langchain Tools."""
    internal_data_tool = Tool(
        name="InternalFinancialDataLookup",
        func=_get_internal_financial_data,
        description="Use ONLY to retrieve internal financial data points ('Current Assets', 'Current Liabilities', 'Inventory', 'Cash'). Input MUST specify metric, unit (e.g., 'Unit Alpha'), and period (e.g., '2025Q1'). Example: 'Get Inventory for Unit Beta for 2024Q4'."
    )
    public_data_tool = Tool(
        name="PublicMarketDataLookup",
        func=_get_public_market_data,
        description="Use to get public market info or industry benchmarks ('tech industry', 'manufacturing industry'). Can ask for specific benchmark ratios. Example: 'Get tech industry benchmark for Average Current Ratio'."
    )
    calculator_tool = Tool(
        name="LiquidityRatioCalculator",
        func=_calculate_liquidity_ratio,
        description="Use ONLY AFTER retrieving ALL necessary values. Input MUST be a dictionary with 'ratio_to_calculate' ('Current Ratio' or 'Quick Ratio') AND numeric values for required fields (e.g., 'Current Assets', 'Current Liabilities', 'Inventory'). Example input dict: {'ratio_to_calculate': 'Quick Ratio', 'Current Assets': 1200000, 'Inventory': 400000, 'Current Liabilities': 800000}."
    )
    return [internal_data_tool, public_data_tool, calculator_tool]

# --- LLM and Agent Setup ---

def initialize_llm(api_key: str, fine_tuned_model_id: str = None):
    """Initializes the ChatOpenAI model, preferring fine-tuned ID if provided."""
    model_to_use = fine_tuned_model_id or DEFAULT_CHAT_MODEL
    try:
        print(f"Attempting to use LLM: {model_to_use}")
        llm = ChatOpenAI(model_name=model_to_use, temperature=0, openai_api_key=api_key)
        print(f"Successfully initialized LLM: {model_to_use}")
        return llm
    except Exception as e_primary:
        print(f"Warning: Failed to initialize LLM '{model_to_use}'. Error: {e_primary}")
        if model_to_use != FALLBACK_CHAT_MODEL:
            print(f"Attempting to fall back to {FALLBACK_CHAT_MODEL}...")
            try:
                llm = ChatOpenAI(model_name=FALLBACK_CHAT_MODEL, temperature=0, openai_api_key=api_key)
                print(f"Successfully initialized fallback LLM: {FALLBACK_CHAT_MODEL}")
                return llm
            except Exception as e_fallback:
                print(f"Error: Failed to initialize fallback LLM '{FALLBACK_CHAT_MODEL}'. Error: {e_fallback}")
                raise ConnectionError("Could not initialize any OpenAI LLM.") from e_fallback
        else:
            raise ConnectionError(f"Could not initialize specified LLM '{model_to_use}'.") from e_primary


def setup_agent_executor(llm, tools: list[Tool]) -> AgentExecutor:
    """Sets up the Langchain agent and executor."""
    prompt = hub.pull("hwchase17/react") # Pull standard ReAct prompt
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    print("Agent Executor setup complete.")
    return agent_executor

# --- Interaction Loop ---

def run_assistant_interaction(agent_executor: AgentExecutor):
    """Runs the interactive command-line loop for the assistant."""
    print("\n--- Liquidity Analysis Assistant ---")
    print("Ask questions about liquidity for Unit Alpha/Beta (e.g., 2025Q1, 2024Q4) or industry benchmarks.")
    print("Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_query = input("\nYour Question: ")
            if user_query.lower() in ["exit", "quit"]:
                break
            if not user_query:
                continue

            response = agent_executor.invoke({"input": user_query})
            print("\nAssistant:")
            print(response['output'])

        except Exception as e:
            print(f"\nAn error occurred during interaction: {e}")
            # Add more robust error handling / recovery if needed
        except KeyboardInterrupt:
            print("\nExiting due to user interrupt.")
            break

    print("\nAssistant session ended.")