from typing import Union
from langchain.agents import initialize_agent, AgentType
import json

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_ap_ar_agent.utils.invoice_parser_util import extract_text_from_pdf, extract_text_from_image

llm = OpenAI(temperature=0, openai_api_key="your-api-key")

# Define the prompt template for dynamic field mapping
prompt_template = """
Given the following extracted invoice data, map it to the following fields:
- invoice_id
- vendor
- date
- amount
- purchase_order
- payment_method
- extracted_text

Extract the appropriate values for each field.

Extracted invoice data:
{invoice_text}

Output as JSON with the mapped fields.
"""

# Create the prompt
prompt = PromptTemplate(input_variables=["invoice_text"], template=prompt_template)
# Function to map raw invoice text to the schema
def map_invoice_to_schema(invoice_text: str) -> dict:
    # Prepare the prompt with the extracted text
    mapped_output = llm(prompt.format(invoice_text=invoice_text))

    # Convert LLM response into JSON (if valid)
    try:
        mapped_data = json.loads(mapped_output)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse LLM output: {mapped_output}")

    return mapped_data

def process_invoice_input(input_file: Union[str, dict], is_file: bool = False) -> dict:
    if is_file:
        if input_file.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(input_file)
        elif input_file.endswith(('.jpg', '.png', '.jpeg')):
            extracted_text = extract_text_from_image(input_file)
        else:
            raise ValueError("Invalid file format. Please provide a PDF or image file.")
    else:
        # Assume JSON input is provided, and we use the extracted text
        extracted_text = input_file.get('extracted_text', "")

    # Map the extracted text to the invoice schema using the LLM
    mapped_invoice_data = map_invoice_to_schema(extracted_text)

    # Return the mapped data as a dictionary
    return mapped_invoice_data

def run_llm_invoice_agent(query: str, input_data: Union[str, dict]) -> str:
    # Process the input (either file or JSON)
    invoice_data = process_invoice_input(input_data, is_file=False)  # Assume mock JSON input for this example

    # Setup LLM and agent tools
    tools = [
        {"name": "invoice_verification", "func": lambda data: run_invoice_verification(data), "description": "Verify the invoice"},
        {"name": "po_matching", "func": lambda data: run_po_matching(data), "description": "Match the Purchase Order"},
        {"name": "approval_process", "func": lambda data: run_approval_process(data), "description": "Process approval"},
        {"name": "payment_processing", "func": lambda data: run_payment_processing(data), "description": "Handle payment processing"}
    ]

    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Handle specific queries or generic ones
    if "analyze this invoice" in query.lower():
        results = []
        for tool in tools:
            results.append(agent.run(query))  # Run each tool in sequence for generic queries
        return "\n".join(results)

    # For specific queries (e.g., 'What is the approval process?')
    return agent.run(query)
