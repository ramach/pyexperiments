from datetime import datetime

import streamlit as st
import json
import os
import sys
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf
from utils.text_extraction import extract_text_from_image
from utils.text_extraction import extract_business_rules_from_docx
from agents.llm_invoice_agent import map_extracted_text_to_invoice_data_with_confidence_score

def process_upload(file):
    if file is None:
        return None
    if file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.lower().endswith((".png", ".jpg")):
        return extract_text_from_image(file)
    elif file.name.lower().endswith(".docx"):
        return extract_business_rules_from_docx(file)
    return None

# Business logic
def validate_invoice(invoice: dict, rules: dict) -> dict:
    errors = []
    logger.debug(f"[validate_invoice], {invoice}")
    # Use rules for dynamic thresholds
    max_hourly_rate = rules.get('max_hourly_rate', 141)
    max_total_hours = rules.get('max_total_hours', 1976)
    max_weekly_hours = rules.get('max_weekly_hours', 40)
    payment_terms_days = rules.get('payment_terms_days', 30)

    if invoice['hourly_rate'] > max_hourly_rate:
        errors.append(f"Hourly rate exceeds ${max_hourly_rate}/hr.")
    if invoice['total_hours'] > max_total_hours:
        errors.append(f"Total hours exceed {max_total_hours}.")
    if invoice['weekly_hours'] > max_weekly_hours and not invoice.get('prior_approval', False):
        errors.append(f"More than {max_weekly_hours} hours/week without prior approval.")
    if not invoice.get('timesheet_approved', False):
        errors.append("Timesheet is not client-approved.")
    if invoice['expenses'] > 0 and not invoice.get('receipts_attached', False):
        errors.append("Receipts missing for submitted expenses.")

    invoice_date = datetime.strptime(invoice['invoice_date'], '%Y-%m-%d')
    last_friday = datetime.strptime(invoice['last_friday_of_month'], '%Y-%m-%d')
    if abs((invoice_date - last_friday).days) > 5:
        errors.append("Invoice date is not close to last Friday of month.")
    if invoice.get('payment_terms_days', 30) > payment_terms_days:
        errors.append(f"Payment terms exceed {payment_terms_days} days.")

    return {
        "valid": len(errors) == 0,
        "errors": errors or ["Invoice meets all rules."]
    }

# LangChain tool
def validate_invoice_tool(input_data: str) -> str:
    """
    Validates an invoice against predefined business rules.
    Expects a JSON string with keys: invoice (dict) and rules (dict).
    """
    try:
        data = json.loads(input_data)
        invoice = data['invoice']
        rules = data['rules']
        result_invoice = validate_invoice(invoice, rules)
        return json.dumps(result_invoice, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# LangChain agent
tools =  [Tool(
    name="invoice_verification",
    func=validate_invoice_tool,
    description="Verify invoice details (e.g., ID, vendor, amount)"
)]

agent = initialize_agent(
    tools=tools,
    llm = ChatOpenAI(temperature=0, model_name="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("🧾 Invoice Validation Tool (with File Upload)")

st.write("Upload your **invoice** and **rules** JSON files below.")

#invoice_file = st.file_uploader("Upload Invoice JSON", type=["json"])
invoice_file = st.file_uploader("Upload Invoice (PDF or image)", type=["pdf", "png", "jpg"])
invoice_text = process_upload(invoice_file)
rules_file = st.file_uploader("Upload Rules JSON", type=["json"])

if st.button("✅ Validate Invoice"):
    if invoice_file and rules_file:
        try:
            rules_data = json.load(rules_file)
            invoice_data = map_extracted_text_to_invoice_data_with_confidence_score(invoice_text) if invoice_text else None
            st.subheader("📌 Extracted Fields")
            st.code(invoice_data, language="json")
            st.json(invoice_data)
            input_data = json.dumps({
                "invoice": invoice_data,
                "rules": rules_data
            })
            with st.spinner("Validating..."):
                result = agent.run(f"Validate this invoice: {input_data}")
                st.success("✅ Validation Completed!")
                st.subheader("🔍 Agent Output")
                st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload both Invoice and Rules JSON files.")
