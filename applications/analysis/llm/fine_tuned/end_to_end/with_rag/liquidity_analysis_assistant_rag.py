# assistant_logic.py
from __future__ import annotations

import re
import os # Added os import
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Added OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.vectorstores import Chroma # Added Chroma
from langchain.schema import Document # Added Document

# Import dummy data from config (or real data access methods)
from config import (
    DUMMY_INTERNAL_DATA,
    DUMMY_PUBLIC_DATA,
    DEFAULT_CHAT_MODEL,
    FALLBACK_CHAT_MODEL,
    CHROMA_DB_PATH, # Assuming you add CHROMA_DB_PATH = "./chroma_db_liquidity" to config.py
    COLLECTION_NAME # Assuming you add COLLECTION_NAME = "liquidity_docs" to config.py
)

# --- Embedding Function Initialization ---
# Global variable for embedding function to avoid reinitialization
embedding_function = None

def initialize_embeddings(api_key: str):
    """Initializes the embedding function if not already done."""
    global embedding_function
    if embedding_function is None:
        try:
            embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
            print("Embedding function initialized.")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            raise ConnectionError("Could not initialize OpenAI Embeddings.")
    return embedding_function

# --- Vector Store Retrieval Setup ---
# Global variable for retriever
vector_retriever = None

def get_vector_store_retriever(db_path: str, collection_name: str, embeddings):
    """Loads the existing ChromaDB store and returns a retriever."""
    global vector_retriever
    if vector_retriever is None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB path not found: {db_path}. Please run build_vector_store.py first.")
        try:
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            # Configure retriever (e.g., k=3 to get top 3 results)
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            print(f"ChromaDB retriever loaded for collection '{collection_name}'.")
        except Exception as e:
            print(f"Error loading vector store from {db_path}: {e}")
            raise ConnectionError("Could not load vector store.") from e
    return vector_retriever


# --- Tool Functions ---

# ... (Keep _get_internal_financial_data, _get_public_market_data, _calculate_liquidity_ratio as before) ...
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

def _retrieve_document_context(query: str) -> str:
    """
    Retrieves relevant context from indexed documents based on the query.
    Use this for questions about analysis, reasons, summaries, or information
    likely found in reports (e.g., 'Why did inventory increase?',
    'Summarize liquidity commentary from latest report').
    """
    print(f"\n--- TOOL CALL: DocumentContextRetriever ---")
    print(f"Query: {query}")
    global vector_retriever
    if vector_retriever is None:
        return "Error: Document retriever is not initialized."
    try:
        # Use the retriever to find relevant documents
        results: list[Document] = vector_retriever.invoke(query)
        if not results:
            return "No relevant context found in documents for this query."

        # Format results for the LLM
        context_str = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in results])
        print(f"Retrieved Context:\n{context_str[:500]}...") # Print snippet
        return context_str
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return f"Error retrieving document context: {e}"

# --- Tool Creation ---

def create_tools(retriever) -> list[Tool]: # Accept retriever as argument
    """Creates and returns a list of Langchain Tools, including the retriever."""
    global vector_retriever # Set the global retriever used by the tool function
    vector_retriever = retriever

    internal_data_tool = Tool(
        name="InternalFinancialDataLookup",
        func=_get_internal_financial_data,
        description="Use ONLY to retrieve specific internal financial data points ('Current Assets', 'Current Liabilities', 'Inventory', 'Cash'). Input MUST specify metric, unit (e.g., 'Unit Alpha'), and period (e.g., '2025Q1'). Example: 'Get Inventory for Unit Beta for 2024Q4'."
    )
    public_data_tool = Tool(
        name="PublicMarketDataLookup",
        func=_get_public_market_data,
        description="Use to get public market info or industry benchmarks ('tech industry', 'manufacturing industry'). Can ask for specific benchmark ratios. Example: 'Get tech industry benchmark for Average Current Ratio'."
    )
    calculator_tool = Tool(
        name="LiquidityRatioCalculator",
        func=_calculate_liquidity_ratio,
        description="Use ONLY AFTER retrieving ALL necessary numerical values. Input MUST be a dictionary with 'ratio_to_calculate' ('Current Ratio' or 'Quick Ratio') AND numeric values for required fields (e.g., 'Current Assets', 'Current Liabilities', 'Inventory'). Example input dict: {'ratio_to_calculate': 'Quick Ratio', 'Current Assets': 1200000, 'Inventory': 400000, 'Current Liabilities': 800000}."
    )
    document_retriever_tool = Tool(
        name="DocumentContextRetriever",
        func=_retrieve_document_context,
        description="Use this tool to find information, explanations, analysis, or commentary within internal reports or documents. Useful for questions asking 'why', requesting summaries from reports, or seeking qualitative information not available as specific data points. Example query: 'What did the Q4 report say about cash flow generation?' or 'Why did receivables increase in 2025Q1?'"
    )
    return [internal_data_tool, public_data_tool, calculator_tool, document_retriever_tool] # Add the new tool

# --- LLM and Agent Setup ---
# (initialize_llm, setup_agent_executor, run_assistant_interaction remain largely the same as before)
# ... (Keep initialize_llm as before) ...
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
        handle_parsing_errors="Check your output and make sure it conforms!", # More specific error handling message
        max_iterations=7 # Allow slightly more iterations if RAG is involved
    )
    print("Agent Executor setup complete.")
    return agent_executor

def run_assistant_interaction(agent_executor: AgentExecutor):
    """Runs the interactive command-line loop for the assistant."""
    print("\n--- Liquidity Analysis Assistant (with Document Search) ---")
    print("Ask questions about liquidity ratios, benchmarks, or insights from indexed documents.")
    print("Examples:")
    print(" - What is the Current Ratio for Unit Alpha in 2025Q1?")
    print(" - Why did inventory increase for Unit Beta last quarter? (Uses Document Search)")
    print(" - Summarize the liquidity commentary from the latest report. (Uses Document Search)")
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
        except KeyboardInterrupt:
            print("\nExiting due to user interrupt.")
            break

    print("\nAssistant session ended.")