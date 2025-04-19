import sys
import os
from typing import Optional

from dotenv import load_dotenv

#Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

load_dotenv()

from pydantic import BaseModel
from pydantic import ValidationError
from tools.invoice_tools_enhanced import (
    run_invoice_verification,
    run_po_matching,
    run_approval_process,
    run_payment_processing
)
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InvoiceInput(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    date: str
    purchase_order: str
    payment_method: str
    extracted_text: Optional[str] = ""

def get_invoice_agent(tools, prompt):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    logger.debug("Initializing invoice agent with tools: %s", [tool.name for tool in tools])
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )

def run_invoice_agent_with_function_calling(query: str, json_input: str):
    try:
        # Parse the invoice data from the JSON string
        invoice_data = json.loads(json_input)
        logger.debug("Initializing invoice agent with: %s", json_input)
        # Initialize the agent's tools
        tools = [
            Tool(name="invoice_verification", func=run_invoice_verification, description="Verify the invoice details"),
            Tool(name="po_matching", func=run_po_matching, description="Match Purchase Order with invoice"),
            Tool(name="approval_process", func=run_approval_process, description="Process invoice approval"),
            Tool(name="payment_processing", func=run_payment_processing, description="Process payment for the invoice")
        ]

        # Setup the LLM chain for the agent
        prompt_template = """
        Given the following invoice data, answer the query:

        Invoice Data:
        {invoice_data}

        Query:
        {query}

        Select the relevant tool to run based on the query.
        """

        # Creating a prompt with the invoice data and query
        prompt = PromptTemplate(input_variables=["invoice_data", "query"], template=prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        llm_chain = LLMChain(llm, prompt=prompt)

        # Create an agent with the tools
        agent = initialize_agent(tools, llm_chain, verbose=True)

        # Log the input for debugging purposes
        logger.info(f"[InvoiceAgent] Running agent with query: {query}")
        logger.debug(f"[InvoiceAgent] Parsed invoice data: {invoice_data}")

        # Run the agent with the provided query and invoice data
        agent_result = agent.run({
            "invoice_data": invoice_data,  # Pass the parsed invoice data
            "query": query  # Pass the query to the agent
        })

        # Log the result and return it
        logger.info(f"[InvoiceAgent] Result: {agent_result}")
        return agent_result

    except Exception as e:
        # Log the error if something goes wrong
        logger.error(f"[InvoiceAgent] Error processing query: {str(e)}")
        return {"error": str(e)}