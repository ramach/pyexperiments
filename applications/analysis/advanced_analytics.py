import pandas as pd
import os
import io
import requests
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode


def liquidity_analysis_cloud(url, domain_filter=None):
    """
    Performs liquidity analysis on companies from a CSV file hosted at a URL.

    Args:
        url (str): The URL of the CSV file.  Must be publicly accessible.
        domain_filter (str, optional):  If provided, only companies with this
            'act_symbol' will be analyzed.  Case-insensitive.

    Returns:
        pandas.DataFrame: A DataFrame containing the liquidity analysis results.
            Returns an empty DataFrame if there's an error.
    """

    try:
        # Download the CSV data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        csv_data = io.StringIO(response.text)

        # Read the CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_data)
        print(df.columns())
        # --- Data Cleaning and Preparation ---

        # Check for required columns
        required_columns = [
            "act_symbol",
            "date",
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing required columns.  Must have: {required_columns}")
            return pd.DataFrame()

        # Convert relevant columns to numeric (handling potential errors)
        numeric_columns = [
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts errors to NaN

        # Convert 'date' to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with any NaN values in the critical columns (after conversion)
        df.dropna(subset=numeric_columns + ['date'], inplace=True)

        # --- Filtering by Domain (act_symbol) ---

        if domain_filter:
            df = df[df['act_symbol'].str.lower() == domain_filter.lower()]

        if df.empty:
            print(f"No data found for domain filter: {domain_filter}")
            return pd.DataFrame()

        # --- Liquidity Calculations ---

        # 1. Current Ratio
        df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities']

        # 2. Quick Ratio (Acid-Test Ratio)
        df['quick_ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liabilities']

        # 3. Cash Ratio
        df['cash_ratio'] = df['cash_and_equivalents'] / df['total_current_liabilities']

        # 4. Inventory to Current Assets Ratio
        df['inventory_to_current_assets'] = df['inventories'] / df['total_current_assets']

        # --- Results ---
        # Sort by date for easier interpretation
        results_df = df.sort_values(['act_symbol', 'date'])

        # Select relevant columns for the output
        results_df = results_df[[
            'act_symbol',
            'date',
            'total_current_assets',
            'total_current_liabilities',
            'inventories',
            'cash_and_equivalents',
            'current_ratio',
            'quick_ratio',
            'cash_ratio',
            'inventory_to_current_assets'
        ]]
        return results_df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def liquidity_analysis(url, domain_filter=None):
    """
    Performs liquidity analysis on companies from a CSV file hosted at a URL.

    Args:
        url (str): The URL of the CSV file.  Must be publicly accessible.
        domain_filter (str, optional):  If provided, only companies with this
            'act_symbol' will be analyzed.  Case-insensitive.

    Returns:
        pandas.DataFrame: A DataFrame containing the liquidity analysis results.
            Returns an empty DataFrame if there's an error.
    """

    try:
        # Read the CSV data into a Pandas DataFrame
        df = pd.read_csv(
            "/Users/krishnaramachandran/Downloads/post-no-preference_earnings_master_balance_sheet_assets.csv",
            on_bad_lines='skip')
        print(df.columns())
        # --- Data Cleaning and Preparation ---

        # Check for required columns
        required_columns = [
            "act_symbol",
            "date",
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing required columns.  Must have: {required_columns}")
            return pd.DataFrame()

        # Convert relevant columns to numeric (handling potential errors)
        numeric_columns = [
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts errors to NaN

        # Convert 'date' to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with any NaN values in the critical columns (after conversion)
        df.dropna(subset=numeric_columns + ['date'], inplace=True)

        # --- Filtering by Domain (act_symbol) ---

        if domain_filter:
            df = df[df['act_symbol'].str.lower() == domain_filter.lower()]

        if df.empty:
            print(f"No data found for domain filter: {domain_filter}")
            return pd.DataFrame()

        # --- Liquidity Calculations ---

        # 1. Current Ratio
        df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities']

        # 2. Quick Ratio (Acid-Test Ratio)
        df['quick_ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liabilities']

        # 3. Cash Ratio
        df['cash_ratio'] = df['cash_and_equivalents'] / df['total_current_liabilities']

        # 4. Inventory to Current Assets Ratio
        df['inventory_to_current_assets'] = df['inventories'] / df['total_current_assets']

        # --- Results ---
        # Sort by date for easier interpretation
        results_df = df.sort_values(['act_symbol', 'date'])

        # Select relevant columns for the output
        results_df = results_df[[
            'act_symbol',
            'date',
            'total_current_assets',
            'total_current_liabilities',
            'inventories',
            'cash_and_equivalents',
            'current_ratio',
            'quick_ratio',
            'cash_ratio',
            'inventory_to_current_assets'
        ]]
        return results_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


# --- Example Usage ---

# Replace with the actual URL (make sure it's publicly accessible)
csv_url = "https://drive.google.com/file/d/1FFZAPqnhDMv1RqMBjwH83cwb9dt8tWfE/view?usp=sharing"

# 1. Analyze all companies:
all_companies_analysis = liquidity_analysis(csv_url)
if not all_companies_analysis.empty:
    print("\nLiquidity Analysis (All Companies):")
    print(all_companies_analysis)

# 2. Analyze a specific domain (e.g., 'AAPL'):
apple_analysis = liquidity_analysis(csv_url, domain_filter="AAPL")
if not apple_analysis.empty:
    print("\nLiquidity Analysis (AAPL only):")
    print(apple_analysis)

# Example: Analyzing another company (if present in the data)
microsoft_analysis = liquidity_analysis(csv_url, domain_filter='MSFT')
if not microsoft_analysis.empty:
    print("\nLiquidity Analysis for MSFT")
    print(microsoft_analysis)

# Get list of unique companies
# unique_companies = all_companies_analysis['act_symbol'].unique()
# print(f"\nUnique company symbols: {', '.join(unique_companies)}")

# Example - iterate analysis through companies
'''
for company in unique_companies:
    company_analysis = liquidity_analysis(csv_url, domain_filter=company)
    if not company_analysis.empty:
        print(f"\nLiquidity Analysis for {company}")
        print(company_analysis)
        # Further calculations, plotting, or saving within the loop, if desired
        # Calculate and print the average current ratio for the company
        avg_current_ratio = company_analysis['current_ratio'].mean()
        print(f"\nAverage Current Ratio for {company}: {avg_current_ratio:.2f}")

        # Find the period with the highest cash ratio
        max_cash_ratio_row = company_analysis.loc[company_analysis['cash_ratio'].idxmax()]
        print(f"\nPeriod with Highest Cash Ratio for {company}:")
        print(max_cash_ratio_row)
'''


# --- (The liquidity_analysis function from the previous response) ---
# (Paste the entire liquidity_analysis function here, exactly as before)
def liquidity_analysis(url, domain_filter=None):
    """
    Performs liquidity analysis on companies from a CSV file hosted at a URL.

    Args:
        url (str): The URL of the CSV file.  Must be publicly accessible.
        domain_filter (str, optional):  If provided, only companies with this
            'act_symbol' will be analyzed.  Case-insensitive.

    Returns:
        pandas.DataFrame: A DataFrame containing the liquidity analysis results.
            Returns an empty DataFrame if there's an error.
    """

    try:
        # Download the CSV data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        csv_data = io.StringIO(response.text)

        # Read the CSV data into a Pandas DataFrame
        df = pd.read_csv(csv_data)

        # --- Data Cleaning and Preparation ---

        # Check for required columns
        required_columns = [
            "act_symbol",
            "date",
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing required columns.  Must have: {required_columns}")
            return pd.DataFrame()

        # Convert relevant columns to numeric (handling potential errors)
        numeric_columns = [
            "total_current_assets",
            "total_current_liabilities",
            "inventories",
            "cash_and_equivalents",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts errors to NaN

        # Convert 'date' to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with any NaN values in the critical columns (after conversion)
        df.dropna(subset=numeric_columns + ['date'], inplace=True)

        # --- Filtering by Domain (act_symbol) ---

        if domain_filter:
            df = df[df['act_symbol'].str.lower() == domain_filter.lower()]

        if df.empty:
            print(f"No data found for domain filter: {domain_filter}")
            return pd.DataFrame()

        # --- Liquidity Calculations ---

        # 1. Current Ratio
        df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities']

        # 2. Quick Ratio (Acid-Test Ratio)
        df['quick_ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liabilities']

        # 3. Cash Ratio
        df['cash_ratio'] = df['cash_and_equivalents'] / df['total_current_liabilities']

        # 4. Inventory to Current Assets Ratio
        df['inventory_to_current_assets'] = df['inventories'] / df['total_current_assets']

        # --- Results ---
        # Sort by date for easier interpretation
        results_df = df.sort_values(['act_symbol', 'date'])

        # Select relevant columns for the output
        results_df = results_df[[
            'act_symbol',
            'date',
            'total_current_assets',
            'total_current_liabilities',
            'inventories',
            'cash_and_equivalents',
            'current_ratio',
            'quick_ratio',
            'cash_ratio',
            'inventory_to_current_assets'
        ]]
        return results_df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


# --- LangChain Setup ---

# 1.  Set your OpenAI API key (replace with your actual key)

#os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Best practice

# Or, for Colab / temporary use:
# from google.colab import userdata
<<<<<<< HEAD
# OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
=======

>>>>>>> 8312e81 (preliminary commit)
# 2. Create the LangChain Tool
def analyze_liquidity_wrapper(company_symbol: str) -> list[BaseNode]:
    """
    Analyzes the liquidity of a company based on its symbol.
    Returns a string representation of the analysis results.
    """
    csv_url = "https://drive.google.com/file/d/1FFZAPqnhDMv1RqMBjwH83cwb9dt8tWfE/view?usp=sharing"  #hardcoded for ease
    df_results = liquidity_analysis(csv_url, domain_filter=company_symbol)
    documents = [Document(text=str(row)) for index, row in df_results.iterrows()]
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

liquidity_tool = StructuredTool.from_function(
    func=analyze_liquidity_wrapper,
    name="LiquidityAnalysis",
    description="Useful for analyzing the liquidity of a company based on its stock symbol.  Provides financial ratios like current ratio, quick ratio, and cash ratio."
)


def main():
    # 3.  Choose an LLM (e.g., OpenAI's GPT-3.5 or GPT-4)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # 4. Define the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a financial analyst tasked with comparing the liquidity risks of two companies.
                Use the LiquidityAnalysis tool to get the financial data for each company.
                Compare their current ratios, quick ratios, cash ratios, and inventory to current assets ratios.
                Explain any significant differences and what they suggest about each company's ability to meet its short-term obligations.
                Be concise and focus on the key differences.
                """,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 5. Create the Agent
    tools = [liquidity_tool]  # Our custom tool
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- Example Usage (Queries) ---

    # Example 1: Comparing AAPL and MSFT (assuming MSFT data is in the CSV)
    result = agent_executor.invoke({"input": "What are the key liquidity risks facing AAPL compared to MSFT?"})
    print(result["output"])
    print("------")

    # Example 2:  Focusing on specific ratios
    result = agent_executor.invoke({
                                       "input": "Compare the current ratio and cash ratio of AAPL and MSFT.  Which company appears to have a stronger short-term liquidity position?"})
    print(result["output"])
    print("------")

    # Example 3: If the competitor isn't in the data
    result = agent_executor.invoke({"input": "What are the key liquidity risks facing LC compared to SOFI?"})
    print(result["output"])
    print("------")

    # Example 4: Single Company Analysis
    result = agent_executor.invoke(
        {"input": "Analyze the liquidity risks of AAPL. Focus on any trends in the current ratio."})
    print(result["output"])
    print("------")


if __name__ == "__main__":
    main()
