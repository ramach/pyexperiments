from llama_index.core import VectorStoreIndex, Document
import pandas as pd
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
    data = {
        'date': pd.to_datetime(['2023-12-31', '2022-12-31', '2021-12-31']),
        'total_current_assets': [1000000, 800000, 900000],
        'total_current_liabilities': [500000, 450000, 400000],
        'inventories': [200000, 180000, 150000],
        'net_cash_from_operating_activities': [300000, 250000, 280000],
        'net_income': [400000, 300000, 350000],
        'cash_and_equivalents': [100000, 80000, 90000]
        # ... add other relevant columns from your data
    }
    # Calculate liquidity ratios (as you defined in the mapping)
    df = pd.DataFrame(data)
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
        "How has the company's liquidity position changed over this period?",
        "Are there any concerning trends?",
        "What recommendations can you provide for improving liquidity management?",
    ]
    # Get answers from the agent
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = fintech_analyst.run(prompt)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
