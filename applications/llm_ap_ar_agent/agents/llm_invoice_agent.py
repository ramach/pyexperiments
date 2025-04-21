
import os
import sys
from typing import Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

import json
import logging
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from tools.invoice_tools import run_invoice_verification, run_po_matching, run_approval_process, run_payment_processing
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Tool definitions
tools = [
    Tool(
        name="invoice_verification",
        func=run_invoice_verification,
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
# Input validation model using Pydantic
class InvoiceData(BaseModel):
    invoice_id: str
    vendor: str
    date: str
    amount: float
    purchase_order: str
    payment_method: str
    extracted_text: str

# Create the invoice extraction prompt
invoice_extraction_prompt = PromptTemplate(
    input_variables=["extracted_text"],
    template="""
    You are an AI assistant designed to extract structured invoice data from raw extracted text.

    Please parse the following extracted text and output the details in JSON format:

    Extracted Text:
    {extracted_text}

    The output should include:
    - invoice_id
    - vendor
    - date
    - amount
    - purchase_order
    - payment_method
    
    If a field is not present, say "MISSING". Return a JSON object.
    
    """
)

# Create the LLMChain
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
llm_chain = LLMChain(llm=llm, prompt=invoice_extraction_prompt)

#extracted data with confidence score - experimental used only to extract confidence scores
def map_extracted_text_to_invoice_data_with_confidence_score(extracted_text: str) -> dict:
    model_schema = {"invoice_id", "vendor", "amount", "date", "purchase_order", "payment_method"}
    prompt_template =PromptTemplate(
        input_variables=["extracted_text"],
        template = """ ou are an AI assistant designed to extract structured invoice data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:
         
         Extracted Text:
         {extracted_text}
         
         The output should include:
         - invoice_id
         - vendor
         - date
         - amount
         - purchase_order
         - payment_method
         If a field is not present, say "MISSING". Return a JSON object.
         
         """)

    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        response = chain.run(extracted_text)
        data = json.loads(response)
        logger.debug("[InvoiceAgent_mapping_with_confidence] structured_data: %s", data)
    except Exception as e:
        print(f"Error in mapping extracted text: {e}")
        return {}

    return data

def map_extracted_text_to_invoice_data(extracted_text: str) -> dict:
    """
    Maps the extracted text from the invoice to a structured data model.
    Uses an LLM to parse and return a structured JSON object.
    """
    try:
        # Pass the extracted text to the LLM chain to extract structured data
        structured_data = llm_chain.run(extracted_text)
        logger.debug("[InvoiceAgent_mapping] structured_data: %s", structured_data)
        # If the response is in string format, try to parse it into JSON
        return json.loads(structured_data)  # Convert LLM output (assumed to be in JSON) to dictionary
    except Exception as e:
        print(f"Error in mapping extracted text: {e}")
        return {}

# Now define the main function for the invoice agent
def run_llm_invoice_agent(query: str, extracted_text: str) -> str:
    logger.debug("[InvoiceAgent] extracted_text: %s", extracted_text)
    """
    Run the LLM-based invoice agent with the query and extracted text.
    The agent maps the extracted text to structured data and runs the query through the tools.
    """
    # evaluate mapping confidence
    mapped_data = map_extracted_text_to_invoice_data_with_confidence_score(extracted_text)

    logger.debug("mapped invoice data %s", mapped_data)

    # Map the extracted text to structured data
    input_data = map_extracted_text_to_invoice_data(extracted_text)
    logger.debug("[InvoiceAgent] mapped_text: %s", input_data)

    # Log the structured data to verify
    print("Mapped structured data:", input_data)
    # Proceed with handling the query (you can add more tool calls here)
    agent = get_invoice_agent()
    result = agent.run(f"{query} from {json.dumps(input_data)}")
    return result

