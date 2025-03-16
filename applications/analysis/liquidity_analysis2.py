import os
import getpass

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# Initialize LLM and Embeddings
llm = OpenAI(temperature=0)  # Or your preferred LLM
embeddings = OpenAIEmbeddings()

# --- Synthetic Data ---
# (Replace with your actual data loading and formatting)

#sample data
# Sample balance sheet (expanded with period)
balance_sheet = """
Company | Period | Cash & Equivalents | Accounts Receivable | Inventory | Total Current Assets | Total Assets | Liabilities | Equity
------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------
Your Company | Q1 2023 | 1,500,000 | 750,000 | 300,000 | 2,550,000 | 4,000,000 | 1,500,000 | 2,500,000
Competitor A | Q1 2023 | 800,000 | 400,000 | 150,000 | 1,350,000 | 2,000,000 | 800,000 | 1,200,000
Competitor B | Q1 2023 | 2,000,000 | 1,000,000 | 400,000 | 3,400,000 | 5,000,000 | 2,000,000 | 3,000,000
Your Company | Q2 2023 | 1,700,000 | 800,000 | 350,000 | 2,850,000 | 4,500,000 | 1,600,000 | 2,900,000
Competitor A | Q2 2023 | 900,000 | 450,000 | 175,000 | 1,525,000 | 2,200,000 | 850,000 | 1,350,000
Competitor B | Q2 2023 | 2,200,000 | 1,100,000 | 450,000 | 3,750,000 | 5,500,000 | 2,100,000 | 3,400,000
"""

# Sample cash flow statement (expanded with period)
cash_flow_statement = """
Company | Period | Operating Cash Flow | Investing Cash Flow | Financing Cash Flow | Net Change in Cash
------- | -------- | -------- | -------- | -------- | --------
Your Company | Q1 2023 | 750,000 | -300,000 | 150,000 | 600,000
Competitor A | Q1 2023 | 400,000 | -150,000 | 75,000 | 325,000
Competitor B | Q1 2023 | 1,000,000 | -400,000 | 200,000 | 800,000
Your Company | Q2 2023 | 800,000 | -350,000 | 175,000 | 625,000
Competitor A | Q2 2023 | 450,000 | -175,000 | 87,500 | 362,500
Competitor B | Q2 2023 | 1,100,000 | -450,000 | 225,000 | 875,000
"""

# Sample financial ratios (expanded with period)
financial_ratios = """
Company | Period | Current Ratio | Quick Ratio | Cash Ratio | Debt-to-Equity Ratio
------- | -------- | -------- | -------- | -------- | --------
Your Company | Q1 2023 | 2.71 | 1.80 | 0.60 | 0.60
Competitor A | Q1 2023 | 2.56 | 1.69 | 0.44 | 0.67
Competitor B | Q1 2023 | 2.86 | 2.00 | 0.71 | 0.67
Your Company | Q2 2023 | 2.80 | 1.88 | 0.64 | 0.55
Competitor A | Q2 2023 | 2.65 | 1.77 | 0.47 | 0.63
Competitor B | Q2 2023 | 2.95 | 2.08 | 0.74 | 0.62
"""

# Competitor analysis as a text document
competitor_analysis = """
Your company demonstrates a healthy liquidity position overall, 
but there's room for improvement compared to Competitor B. 

Strengths:
* Your company's current ratio (2.71) indicates a good ability to meet short-term obligations, 
  outperforming Competitor A (2.56).
* Your quick ratio (1.80) is also better than Competitor A (1.69), 
  suggesting a stronger ability to meet immediate obligations using liquid assets.

Weaknesses:
* Competitor B has the strongest liquidity position with a current ratio of 2.86 and a quick ratio of 2.00.
* Your company's cash ratio (0.60) is lower than both competitors, 
  indicating less cash on hand to cover immediate liabilities. This could be a concern 
  in case of unexpected expenses or a downturn in business.

Recommendations:
* Focus on increasing cash reserves by improving cash flow management and exploring 
  options for short-term investments.
* Optimize working capital management to improve efficiency and free up cash. 
  This could involve negotiating better terms with suppliers, improving inventory turnover, 
  and accelerating collections from customers.
* Monitor Competitor B's strategies and performance closely to identify best practices 
  and potential threats.
"""

# Combine data into a list of documents
documents = [balance_sheet, cash_flow_statement, financial_ratios, competitor_analysis]

# --- End of Synthetic Data ---

# Load data into Chroma vector database
# Embed each document individually
doc_embeddings = [embeddings.embed_query(doc) for doc in documents]
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