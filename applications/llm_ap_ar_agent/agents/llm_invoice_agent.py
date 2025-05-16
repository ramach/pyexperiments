from typing import Dict

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
    amount: str
    purchase_order: str
    payment_method: str

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

def map_extracted_text_to_business_rules_data_with_confidence_score(extracted_text: str) -> dict:
    prompt_template =PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured invoice data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}

         The output should include:
         - title
         - version
         - section
         - rules
         each section should have name "name" and its own rules array under "rules"
         please keep title simple
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

#extracted data with confidence score - experimental used only to extract confidence scores
def map_extracted_text_to_invoice_data_with_confidence_score(extracted_text: str) -> dict:
    model_schema = {"invoice_id", "vendor", "amount", "date", "purchase_order", "payment_method"}
    prompt_template =PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured invoice data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}
         
         The output should include:
         - title
         - invoice_id
         - vendor
         - remit_to
         - date
         - amount
         - invoice_title
         - supplier_information
         - period
         - tax
         - insurance
         - payment_method
         - line_items
         if the vendor name is missing use business_id or client id.
         if invoice_title is missing it is "Invoice"
         use "Remit To" field for vendor and vice-versa
         use additional notes for period
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

def map_extracted_text_to_sow_data_with_confidence_score(extracted_text: str) -> dict:
    model_schema = {"invoice_id", "vendor", "amount", "date", "purchase_order", "payment_method"}
    prompt_template =PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured invoice data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}
         
         The output should include:
         - title
         - description
         - term
         - roles_responsibilities
         - additional_terms
         - contractor
         - contacts
         - additional_terms
         - provider_information
         - period
         - line_items
         if the vendor name is missing use business_id or client id.
         for more than one contacts use array and extract each field separately - company, name, phone, email
         if title is missing it is "Statement of Work"
         use "Remit To" field for vendor and vice-versa
         use additional notes for period
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


def map_extracted_text_to_po_data_with_confidence_score(extracted_text: str) -> dict:
    model_schema = {"invoice_id", "vendor", "amount", "date", "purchase_order", "payment_method"}
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured purchase data from raw extracted text.
         for amount field please sum all line_items unit_price multiplied by number of units 
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}
         
         The output should include:
         - title
         - supplier/vendor
         - customer/buyer
         - date
         - amount
         - scope
         - total_value
         - Delivery_Date
         - Schedule
         - rate
         - payment_terms
         - terms
         - max_authorized_amount
         - purchase_order_id
         - payment_method
         - project_description
         - line_items
         if the vendor name is missing use business_id or client id.
         for amount field please sum all line_items unit_price multiplied by number of units.
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

def map_extracted_text_to_timecard_data_with_confidence_score(extracted_text: str) -> dict:
    model_schema = {"contractor_name", "manager_name", "amount", "date_range", "hours_worked", "rate_per_hour", "employee_details"}
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured purchase data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}
         
         The output should include:
         - contractor_name
         - manager_name
         - hours_worked
         - date_range
         - rate
         - amount
         - employee_details
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
    Use LLM to map extracted text to structured invoice fields.
    """
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured purchase data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}
         
         The output should include:
         - vendor
         - invoice_id
         - date
         - amount
         - purchase_order
         - purchase_order_amount
         - payment_method
         - line_items
         if the vendor name is missing use business_id or client id.
         If a field is not present, say "MISSING". Return a JSON object.
         """)

    # Use your LLM call here (OpenAI, Anthropic, etc.)
    response = llm_chain.run(extracted_text)  # <- replace with your function
    try:
        return json.loads(response)
    except Exception as e:
        return {"error": f"Failed to parse LLM response: {e}", "llm_output": response}


# Now define the main function for the invoice agent
def run_llm_invoice_agent(query: str, extracted_text: str) -> str:
    logger.debug("[InvoiceAgent] extracted_text: %s", extracted_text)
    """
    Run the LLM-based invoice agent with the query and extracted text.
    The agent maps the extracted text to structured data and runs the query through the tools.
    """
    # evaluate mapping confidence
    mapped_data_with_confidence_score= map_extracted_text_to_invoice_data_with_confidence_score(extracted_text)

    logger.debug("mapped invoice data %s", mapped_data_with_confidence_score)

    # Map the extracted text to structured data
    mapped_data = map_extracted_text_to_invoice_data(extracted_text)
    logger.debug("[InvoiceAgent] mapped_text: %s", mapped_data)
    validated = InvoiceData(**mapped_data)
    # Log the structured data to verify
    print("Mapped structured data:", json.dumps(validated.dict()))
    # Proceed with handling the query (you can add more tool calls here)
    agent = get_invoice_agent()
    result = agent.run(f"Verify invoice from {json.dumps(validated.dict())}")
    return result

def extract_contract_fields_with_llm(contract_text: str) -> Dict:
    from langchain.prompts import PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template = f""" you are an AI assistant designed to extract structured contract from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {contract_text}
         
         The output should include:
         - Client Name
         - Client Address
         - Consulting Firm Name
         - Consulting Firm Address
         - Scope of Work
         - Fees and Payment Terms
         - Terms and Termination
         if the vendor name is missing use business_id or client id.
         If a field is not present, say "MISSING". Return a JSON object.
         """)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(contract_text)

    try:
        return eval(response)  # Replace with `json.loads()` if result is a valid JSON string
    except Exception as e:
        return {"error": f"Failed to parse result: {e}", "raw_result": response}