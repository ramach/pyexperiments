import os

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Initialize LLM and Embeddings
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0)  # Or your preferred LLM
embeddings = OpenAIEmbeddings()

# --- Synthetic Data ---
# (Replace with your actual data loading and formatting)

# Sample financial statements
balance_sheet = """
Company | Cash | Accounts Receivable | Inventory | Total Assets | Liabilities | Equity
------- | -------- | -------- | -------- | -------- | -------- | --------
Your Company | 1000000 | 500000 | 200000 | 1700000 | 700000 | 1000000
Competitor A | 500000 | 300000 | 100000 | 900000 | 400000 | 500000
Competitor B | 1500000 | 700000 | 300000 | 2500000 | 1000000 | 1500000
"""

cash_flow_statement = """
Company | Operating Cash Flow | Investing Cash Flow | Financing Cash Flow
------- | -------- | -------- | --------
Your Company | 500000 | -200000 | 100000
Competitor A | 300000 | -100000 | 50000
Competitor B | 700000 | -300000 | 150000
"""

# Sample financial ratios
financial_ratios = """
Company | Current Ratio | Quick Ratio | Cash Ratio
------- | -------- | -------- | --------
Your Company | 2.43 | 1.71 | 0.57
Competitor A | 2.25 | 1.5 | 0.33
Competitor B | 2.5 | 1.86 | 0.67
"""

# Sample competitor analysis
competitor_analysis = """
Your company is performing well in terms of liquidity compared to Competitor A, but Competitor B has a stronger liquidity position.
Your company's cash ratio is a weakness, indicating lower cash on hand to cover immediate liabilities.
Focus on increasing cash reserves and optimizing working capital management to improve your liquidity position.
"""

# Combine data into a list of documents
documents = [balance_sheet, cash_flow_statement, financial_ratios, competitor_analysis]

# --- End of Synthetic Data ---

# Load data into Chroma vector database
db = Chroma.from_texts(documents, embeddings, collection_name="enterprise_data")

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# --- Main execution block ---
if __name__ == "__main__":
    # Chatbot interaction loop
    while True:
        query = input("CFO: ")

        # Get response from RetrievalQA chain
        response = qa_chain.run(query)

        # Print response in chat format
        print("Chatbot:", response)

        # Exit condition
        if query.lower() in ["exit", "quit"]:
            break

