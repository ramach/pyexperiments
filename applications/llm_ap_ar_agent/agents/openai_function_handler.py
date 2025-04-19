# agents/openai_function_handler.py
# agents/openai_function_handler.py
from langchain.tools import Tool
from chains.ap_chain import run_ap_chain
from chains.ar_chain import run_ar_chain
from chains.retriever_qa_chain import run_retriever_query
from retrievers.serpapi_scraper import run_serpapi_scraper


def get_openai_tools():
    """Dynamically build tools to avoid LLMChain import-time instantiation issues."""

    def ap_chain_tool(input: str) -> str:
        """Query the Accounts Payable chain with an input string."""
        return run_ap_chain(input)

    def ar_chain_tool(input: str) -> str:
        """Query the Accounts Receivable chain with an input string."""
        return run_ar_chain(input)

    def serpapi_tool(query: str) -> str:
        """Fetch public vendor or enterprise info using SERP API."""
        return run_serpapi_scraper(query)

    def vector_qa_tool(query: str) -> str:
        """Run a GPT-4 powered semantic search over internal data."""
        return run_retriever_query(query)

    return [
        Tool(name="AccountsPayableQuery", func=ap_chain_tool, description="Query AP documents."),
        Tool(name="AccountsReceivableQuery", func=ar_chain_tool, description="Query AR documents."),
        Tool(name="PublicDataScraper", func=serpapi_tool, description="Fetch public data from vendor sites."),
        Tool(name="VectorQASearch", func=vector_qa_tool, description="Query semantic search index over documents."),
    ]