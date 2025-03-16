from datetime import date

from llama_index.core import VectorStoreIndex, Document
import pandas as pd
import numpy as np

from llama_index.core.node_parser import SimpleNodeParser
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
def create_fintech_analyst2(df):
    os.environ["OPENAI_API_KEY"] = ""

    # 1. Load financial data (replace with your actual data sources)
    # Convert DataFrame rows to Document objects
    documents = [Document(text=str(row)) for index, row in df.iterrows()]
    # 2. Initialize LLM
    #Settings.llm = OpenAI(temperature=0, model_name="text-davinci-003")
    #llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # 3. Build the index
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, llm=llm)

    # 4. (Optional) Persist the index for future use
    '''
    storage_context = StorageContext.from_defaults()
    index.storage_context = storage_context
    index.save_to_disk('index.json')
    '''

    query_engine = index.as_query_engine()
    # 5. Define a tool for the agent to interact with the index
    def query_index(query_str):
        """Queries the LlamaIndex."""
        response = query_engine.query(query_str)
        return str(response)

    tools = [
        Tool(
            name="Fintech Analyst",
            func=query_index,
            description="Useful for answering questions about company financials, "
                        "such as liquidity, profitability, and risk analysis. "
                        "Input should be a fully formed question."
        )
    ]

    # 6. Initialize the LangChain agent
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return agent

def main():
    schema = {
        'date': date,  # Date
        'total_current_assets': int,  # Total Current Assets
        'total_current_liabilities': int,  # Total Current Liabilities
        'inventories': int,  # Inventories
        'net_cash_from_operating_activities': int,  # Net Cash from Operating Activities
        'net_income': int,  # Net Income
        'cash_and_equivalents': int,  # Cash and Equivalents
        'accounts_receivable': int,  # Accounts Receivable
        'short_term_debt': int,  # Short-Term Debt
        'long_term_debt': int # Long-Term Debt
        # ... add more columns as needed
    }

    num_data_points = 100

# Generate synthetic data
    df = pd.DataFrame(columns=schema.keys(), index=range(num_data_points))  # Add index here
    df['date'] = pd.to_datetime(np.random.choice(pd.date_range('2018-01-01', '2023-12-31'), num_data_points))
    df['total_current_assets'] = np.random.randint(500000, 2000000, num_data_points)
    df['total_current_liabilities'] = np.random.randint(200000, 800000, num_data_points)
    df['inventories'] = np.random.randint(100000, 500000, num_data_points)
    df['net_cash_from_operating_activities'] = np.random.randint(100000, 500000, num_data_points)
    df['net_income'] = np.random.randint(200000, 700000, num_data_points)
    df['cash_and_equivalents'] = np.random.randint(50000, 200000, num_data_points)
    df['accounts_receivable'] = np.random.randint(100000, 400000, num_data_points)
    df['short_term_debt'] = np.random.randint(100000, 300000, num_data_points)
    df['long_term_debt'] = np.random.randint(300000, 1000000, num_data_points)

    # Calculate liquidity ratios (as you defined in the mapping)
    df['Working Capital'] = df['total_current_assets'] - df['total_current_liabilities']
    df['Current Ratio'] = df['total_current_assets'] / df['total_current_liabilities']
    df['Quick Ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liabilities']
    df['Cash Ratio'] = df['cash_and_equivalents'] / df['total_current_liabilities']
    df['Operating Cash Flow Ratio'] = df['net_cash_from_operating_activities'] / df['total_current_liabilities']
    df['Cash Flow to Revenue Ratio'] = df['net_cash_from_operating_activities'] / df['net_income']
    fintech_analyst = create_fintech_analyst2(df)
    # Example prompts from treasury and CFO
    prompts = [
        "What is the company's current ratio?",
        "How has the company's cash flow changed over the past year?",
        "What are the key liquidity risks facing the company?",
        "Can you provide a summary of the company's financial performance?",
    ]
    # Get answers from the agent
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = fintech_analyst.run(prompt)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()