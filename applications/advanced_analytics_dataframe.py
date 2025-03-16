import getpass
import os
import pandas as pd

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


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
        df = pd.read_csv(url, on_bad_lines='skip')
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

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    df = liquidity_analysis("/Users/krishnaramachandran/Downloads/post-no-preference_earnings_master_balance_sheet_assets.csv", 'LC')
    df_cash_flow = pd.read_csv("/Users/krishnaramachandran/Downloads/cash_flow_statement_lc.csv.csv")
    pd.set_option('display.max_columns', None)
    print(df.head)
    competitor_analysis = """
Your company demonstrates a healthy liquidity position overall,
but there's room for improvement compared to Competitor B.

Strengths:
* Your company's current ratio (2.71 in Q1 2023, 2.80 in Q2 2023) 
  indicates a good ability to meet short-term obligations, outperforming 
  Competitor A (2.56 in Q1 2023, 2.65 in Q2 2023).
* Your quick ratio (1.80 in Q1 2023, 1.88 in Q2 2023) is also better 
  than Competitor A (1.69 in Q1 2023, 1.77 in Q2 2023), suggesting a 
  stronger ability to meet immediate obligations using liquid assets.

Weaknesses:
* Competitor B has the strongest liquidity position with a current ratio 
  of 2.86 in Q1 2023 and 2.95 in Q2 2023, and a quick ratio of 2.00 in 
  Q1 2023 and 2.08 in Q2 2023.
* Your company's cash ratio (0.60 in Q1 2023, 0.64 in Q2 2023) is 
  lower than both competitors, indicating less cash on hand to cover 
  immediate liabilities. This could be a concern in case of unexpected 
  expenses or a downturn in business.

Recommendations:
* Focus on increasing cash reserves by improving cash flow management 
  and exploring options for short-term investments.
* Optimize working capital management to improve efficiency and free up 
  cash. This could involve negotiating better terms with suppliers, 
  improving inventory turnover, and accelerating collections from customers.
* Monitor Competitor B's strategies and performance closely to identify 
  best practices and potential threats.
"""
    documents = [df.to_string(), df_cash_flow.to_string(), competitor_analysis]
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    db = Chroma.from_texts(documents, embeddings, collection_name="enterprise_data")
    # Initialize RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    while True:
        query = input("CFO: ")

        # Get response from RetrievalQA chain
        response = qa_chain.run(query)

        # Print response in chat format
        print("Chatbot:", response)

        # Exit condition
        if query.lower() in ["exit", "quit"]:
            break

if __name__ == "__main__":
    main()