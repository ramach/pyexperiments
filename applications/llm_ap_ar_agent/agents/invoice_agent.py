# agents/invoice_agent.py
import logging

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from tools.invoice_tools import (
    run_invoice_verification,
    run_approval_process,
    run_po_matching,
    run_payment_processing
)
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import json

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Pydantic input
class InvoiceInput(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    date: str
    purchase_order: str
    payment_method: str


def wrapped_invoice_verification(data):
    try:
        logger.debug(f"[wrapped_invoice_verification] Raw input: {data} ({type(data)})")

        if isinstance(data, dict):
            return run_invoice_verification(json.dumps(data))

        if isinstance(data, str):

            try:
                parsed = json.loads(data)
                return run_invoice_verification(json.dumps(parsed))  # Ensure it's stringified again
            except json.JSONDecodeError:
                return "[Invoice Verification] Error: Expected JSON string but got invalid JSON."

        return "[Invoice Verification] Error: Unsupported input type."

    except Exception as e:
        return f"[wrapped_invoice_verification Exception] {str(e)}"


# Tool definitions
tools = [
    Tool(
        name="invoice_verification",
        func=wrapped_invoice_verification,
        description="Use this tool to verify an invoice. Required fields: invoice_id, vendor, amount, date."
    ),
    Tool(
        name="po_matching",
        func=run_po_matching,
        description="Use this to check if invoice matches the correct purchase order"
    ),
    Tool(
        name="approval_process",
        func=run_approval_process,
        description="Use this to determine who needs to approve the invoice"
    ),
    Tool(
        name="payment_processing",
        func=run_payment_processing,
        description="Use this to process the payment using QuickBooks simulation"
    ),
]


def get_invoice_agent():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    logger.debug("Initializing invoice agent with tools: %s", [tool.name for tool in tools])
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )


def run_invoice_agent_with_function_calling(input: str) -> str:
    try:
        payload = json.loads(input)
        validated = InvoiceInput(**payload)
        logger.debug("[InvoiceAgent] Parsed input: %s", validated.dict())

        agent = get_invoice_agent()

        query = (
            f"Verify invoice from {validated.vendor} dated {validated.date} with amount ${validated.amount} and {validated.invoice_id}"
        )
        logger.info("[InvoiceAgent] Running agent with query: %s", query)

        result = agent.run(f"Verify invoice from {json.dumps(validated.dict())}")
        logger.debug("[InvoiceAgent] Result: %s", result)
        return result
    except Exception as e:
        logger.exception("[InvoiceAgent Error]")
        return f"[InvoiceAgent Error] {str(e)}"
