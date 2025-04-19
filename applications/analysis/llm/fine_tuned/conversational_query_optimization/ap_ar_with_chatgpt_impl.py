# LLM-Based AP/AR Agent with Conversational Query Optimization and Public Data Integration
'''
Here's a full end-to-end implementation of a Python-based LLM agent for handling both Accounts Payable (AP) and Accounts Receivable (AR) tasks. It includes:

Document loading and vectorization

Conversational query optimization

Specialized tools for AP and AR

An LLM-powered agent that dynamically routes and responds

Let me know if you'd like this extended to include public data sources like SEC filings, vendor websites, or email ingestion!






You said:
please extend using public data, vendor websites and mock


ChatGPT said:
Edited
I've extended your implementation to include mock public data sources and vendor website information. This includes:

Mock scraping from vendor websites.

Mock access to public financial data (like SEC-style updates).

Additional tools integrated into the agent for querying this new information.

Let me know if you'd like to replace mock functions with real API integrations (like EDGAR for SEC data or BeautifulSoup-based scraping), or want a UI interface for this agent!






You said:
can you include a driver (main) with suitable arguments to pass


ChatGPT said:
Edited
✅ I've added a main driver using argparse so you can run the agent from the command line with the following arguments:

--query: The user query

--mode: Choose from ap, ar, vendor, or public

--ap_file: Path to accounts payable docs (default: data/accounts_payable.txt)

--ar_file: Path to accounts receivable docs (default: data/accounts_receivable.txt)

Example usage:

bash
Copy
Edit
python your_script.py --query "What invoices are due this week for ABC Ltd?" --mode ap
Let me know if you want to turn this into a REST API or streamlit-style dashboard!






You said:
can you add a tokenizing step so the query optimizer gets the most relevant document


ChatGPT said:
Edited
✅ Added a token counting and filtering step using tiktoken so the QueryOptimizer returns only the most relevant document chunks without exceeding a token limit (default is 1500). Let me know if you'd like this integrated with OpenAI's function calling or into a Streamlit/Gradio interface!






You said:
yes integrate into OpenAI's function calling


ChatGPT said:
Edited

Answer in chat instead
✅ Integrated OpenAI function calling style with a query_financial_assistant method. It's ready for use with APIs or serverless calls—just pass the query and mode (like "ap", "ar", "vendor", or "public").

Want to wrap this in a FastAPI or Flask endpoint next?
'''
import os
import json
import openai
import requests
import argparse
import tiktoken
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from openai import OpenAI

# Initialize OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load and preprocess internal documents (Invoices, Purchase Orders, Payment Records)
def load_internal_docs(doc_path: str):
    loader = TextLoader(doc_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    return split_docs

# Step 2: Scrape and preprocess vendor websites (mock example for demo purposes)
def scrape_vendor_website(vendor_name: str) -> str:
    mock_data = {
        "ABC Ltd": "Vendor ABC Ltd provides monthly invoices on the 1st. Payment terms are Net 30.",
        "XYZ Inc": "XYZ Inc issues invoices biweekly with a 15-day payment window."
    }
    return mock_data.get(vendor_name, "No vendor data available.")

# Step 3: Load public financial data (mock SEC data)
def load_public_financial_data(company: str) -> str:
    public_data = {
        "EnterpriseX": "EnterpriseX quarterly earnings Q1 2025 show a 5% increase in AR turnover.",
        "CompanyY": "CompanyY has delayed vendor payments due to cash flow issues in Q4 2024."
    }
    return public_data.get(company, "No public financial data found.")

# Step 4: Create a FAISS Vector Store
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Step 5: Retrieval QA Chain for AP and AR tasks
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa

# Tokenizer helper
encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    return len(encoding.encode(string))

# Step 6: Tool for optimizing conversational queries to relevant documents
class QueryOptimizer:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever()

    def optimize_query(self, query: str, max_tokens: int = 1500) -> str:
        results = self.retriever.get_relevant_documents(query)
        selected_docs = []
        token_count = 0
        for doc in results:
            doc_tokens = num_tokens_from_string(doc.page_content)
            if token_count + doc_tokens > max_tokens:
                break
            selected_docs.append(doc.page_content)
            token_count += doc_tokens
        return "\n".join(selected_docs)

# OpenAI function calling style
function_schema = {
    "name": "query_financial_assistant",
    "description": "Query financial agent using AP/AR/public data",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User question regarding AP, AR, vendor or public data"
            },
            "mode": {
                "type": "string",
                "enum": ["ap", "ar", "vendor", "public"],
                "description": "Which mode of data the agent should use"
            }
        },
        "required": ["query", "mode"]
    }
}

def query_financial_assistant(query: str, mode: str, qa_chain_ap, qa_chain_ar, query_optimizer_ap, query_optimizer_ar, agent) -> str:
    if mode == "ap":
        context = query_optimizer_ap.optimize_query(query)
    elif mode == "ar":
        context = query_optimizer_ar.optimize_query(query)
    else:
        context = ""
    full_query = f"Context: {context}\n\nQuestion: {query}"
    return agent.run(full_query)

# Global chains for tools
qa_chain_ap = None
qa_chain_ar = None
query_optimizer_ap = None
query_optimizer_ar = None

# Tool functions
def accounts_payable_tool(input: str) -> str:
    return qa_chain_ap.run(input)

def accounts_receivable_tool(input: str) -> str:
    return qa_chain_ar.run(input)

def vendor_data_tool(vendor_name: str) -> str:
    return scrape_vendor_website(vendor_name)

def public_financial_tool(company_name: str) -> str:
    return load_public_financial_data(company_name)

# Step 8: Combine into agent with tools
def create_financial_agent():
    tools = [
        Tool(
            name="Accounts Payable Assistant",
            func=accounts_payable_tool,
            description="Handles vendor-related payable queries like due dates, invoice matching, and status."
        ),
        Tool(
            name="Accounts Receivable Assistant",
            func=accounts_receivable_tool,
            description="Handles enterprise AR queries like outstanding payments, invoice tracking, and customer statements."
        ),
        Tool(
            name="Vendor Public Info",
            func=vendor_data_tool,
            description="Fetches publicly available vendor information."
        ),
        Tool(
            name="Public Financial Info",
            func=public_financial_tool,
            description="Retrieves mock public financial data about companies."
        )
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

# Driver
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based AP/AR Agent CLI")
    parser.add_argument("--query", type=str, required=True, help="User query for the agent")
    parser.add_argument("--mode", type=str, choices=["ap", "ar", "vendor", "public"], default="ap", help="Type of query")
    parser.add_argument("--ap_file", type=str, default="data/accounts_payable.txt", help="Path to AP document")
    parser.add_argument("--ar_file", type=str, default="data/accounts_receivable.txt", help="Path to AR document")
    args = parser.parse_args()

    # Load internal documents
    internal_docs_ap = load_internal_docs(args.ap_file)
    internal_docs_ar = load_internal_docs(args.ar_file)

    # Create vector stores
    vectorstore_ap = create_vector_store(internal_docs_ap)
    vectorstore_ar = create_vector_store(internal_docs_ar)

    # Create QA Chains
    qa_chain_ap = create_qa_chain(vectorstore_ap)
    qa_chain_ar = create_qa_chain(vectorstore_ar)

    # Query Optimizers
    query_optimizer_ap = QueryOptimizer(vectorstore_ap)
    query_optimizer_ar = QueryOptimizer(vectorstore_ar)

    # Create Agent
    financial_agent = create_financial_agent()

    # Function call simulation
    result = query_financial_assistant(
        query=args.query,
        mode=args.mode,
        qa_chain_ap=qa_chain_ap,
        qa_chain_ar=qa_chain_ar,
        query_optimizer_ap=query_optimizer_ap,
        query_optimizer_ar=query_optimizer_ar,
        agent=financial_agent
    )
    print("\n=== Agent Response ===")
    print(result)