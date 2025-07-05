from typing import Dict
import pandas as pd

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

def map_extracted_text_to_msa_data_with_confidence_score(extracted_text: str) -> dict:
    prompt_template =PromptTemplate(
        input_variables=["extracted_text"],
        template = """ you are an AI assistant designed to extract structured invoice data from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {extracted_text}

         The output should include:
         - title
         - summary
         - rules like services
         - rule
         each rule under rules should have a description
         each rule should have name "name" and its own rules array under "sub-sections"
         please make sure all sentences are included
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
         each rule under rules should have a description
         each section should have name "name" and its own rules array under "rules"
         each sentence will be a rule under rules
         please make sure all sentences are included
         each bulleted item can be a rule within that section
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
         - bill_to
         - vendor
         - date
         - amount
         - invoice_title
         - supplier_information
         - period
         - terms
         - tax
         - insurance
         - due_date
         - payment_method
         - additional_text
         - line_items
         if the vendor name is missing use business_id or client id.
         if invoice_title is missing it is "Invoice"
         use "Remit To" field for vendor and vice-versa
         split vendor and bill_to fields into name, address, phone no and email on separate lines
         use additional notes for period
         payment_method should be available as a field else say "MISSING"
         for additional_text please use anything in the document not covered by this schema
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
         - description_of_service
         - term
         - roles_responsibilities
         - additional_terms
         - contractor
         - contracting_company
         - contacts
         - provider_information
         - period
         - maximum_hours
         - maximum_authorized_fee
         - line_items
         if the vendor name is missing use business_id or client id.
         contracting_company is the name of the company for which the contractor works. name of the vendor company
         provider_information is in contacts section. it is the name of the company associated with the contractor. Use contracting_company
         for more than one contacts use array and extract each field separately - company, name, phone, email
         if title is missing it is "Statement of Work"
         use "Remit To" field for vendor and vice-versa
         use additional notes for period
         get contractor from roles_responsibilities
         break description_service into an array of rules for each sentence
         break line_items into an array of rules. One rule for each sentence
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
         all fields that can be extracted including "worker accounting" section
         ignore blank lines.
         Return a JSON object.
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


def map_extracted_text_to_po_data_with_confidence_score_orig(extracted_text: str) -> dict:
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
         - timesheet
         - expense_sheet
         - worker_accounting extract all fields
         - work_order_accounting extract all fields
         - worker
         - posting_information
         - personal_information
         - current_work_order_accounting
         - line_items
         if the vendor name is missing use business_id or client id.
         for amount field please sum all line_items unit_price multiplied by number of units.
         ignore blank lines.
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

def map_excel_to_timecard_data_with_confidence_score(excel_file: str) -> dict:
    extracted_text = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl', usecols='A,B,K,L').get("Apr").to_string()
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
         - date_range
         - rate
         - total_amount
         - client_name
         - employee_details
         - total_hours_worked
         - hours_worked
         total_hours_worked is from "Total"  and total_amount from "Total $"
         add hours worked as array for each day during the date range as an hours_worked in this format day:hours. For missing hours use 0
         If a field is not present, say "MISSING". client_name is second token after Unnamed. Return a JSON object.
         """)

    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        response = chain.run(extracted_text)
        logger.debug("[InvoiceAgent_mapping_with_confidence] structured_data: %s", response)
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
         - date_range
         - rate
         - total_amount
         - client_name
         - employee_details
         - total_hours_worked
         - hours_worked
         total_hours_worked is from "Total"  and total_amount from "Total $"
         add hours worked as array for each day during the date range as an hours_worked in this format day:hours. For missing hours use 0
         If a field is not present, say "MISSING". client_name maybe extracted from the first row - first non-empty word or not equal to Unnamed. Return a JSON object.
         """)

    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        response = chain.run(extracted_text)
        logger.debug("[InvoiceAgent_mapping_with_confidence] structured_data: %s", response)
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
        input_variables=["contract_text"],
        template = f""" you are an AI assistant designed to extract structured contract from raw extracted text.
         Please parse the following extracted text and output the details in JSON format with "value" and "confidence" (0.0 to 1.0) for each field:

         Extracted Text:
         {contract_text}
         
         The output should include:
         - client_name
         - client_address
         - Consulting Firm Name
         - Consulting Firm Address
         - Scope of Work
         - Fees and Payment Terms
         - Terms and Termination
         if the vendor name is missing use business_id or client id.
         If a field is not present, say "MISSING". Return a JSON object.
         """)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"contract_text": contract_text})

    try:
        return eval(response)  # Replace with `json.loads()` if result is a valid JSON string
    except Exception as e:
        return {"error": f"Failed to parse result: {e}", "raw_result": response}