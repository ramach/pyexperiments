import streamlit as st
import json
import os
import sys
from dotenv import load_dotenv


load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.wrapped_llm_invoice_agent import run_llm_invoice_agent, LLMInvoiceAgentInput
from utils.text_extraction import extract_text_from_pdf
from utils.text_extraction import extract_text_from_image
from agents.llm_contract_extractor import extract_text_from_docx, extract_contract_fields_with_llm_experimental
from utils.text_extraction import extract_business_rules_from_docx
from agents.llm_invoice_agent import map_extracted_text_to_invoice_data_with_confidence_score
from agents.llm_invoice_agent import map_extracted_text_to_invoice_data
from agents.llm_invoice_agent import map_extracted_text_to_po_data_with_confidence_score
from agents.llm_invoice_agent import map_extracted_text_to_sow_data_with_confidence_score
from agents.llm_invoice_agent import map_extracted_text_to_business_rules_data_with_confidence_score
from agents.llm_business_rule_agent import map_rule_text_to_structured
from utils.text_extraction import robust_extract_text
from agents.llm_invoice_agent import  map_extracted_text_to_timecard_data_with_confidence_score

st.set_page_config(page_title="LLM Invoice Agent", layout="wide")
st.title("📄 LLM Invoice + PO + Contract + Business Rules Agent")

# Query input
query = st.text_input("Enter your question for the invoice agent", "Analyze this invoice")

# Upload files
invoice_file = st.file_uploader("Upload Invoice (PDF or image)", type=["pdf", "png", "jpg"])
po_file = st.file_uploader("Upload Purchase Order (PDF or image or docx)", type=["pdf", "png", "jpg", "docx"])
contract_file = st.file_uploader("Upload Contract (PDF or image or docx)", type=["pdf", "png", "jpg", "docx"])
rules_file = st.file_uploader("Upload policy file (PDF or image or docx)", type=["pdf", "png", "jpg", "docx"])
statement_of_work_file = st.file_uploader("Upload SOW (DOCX)", type=["docx"])
uploaded_timecard_file_pdf = st.file_uploader("Upload Time Card pdf", type=["pdf"])

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

# Extract and map
invoice_data = None
po_data = None
contract_data = None
rules_data = None
sow_data = None

if st.button("Run Agent"):
    with st.spinner("Processing..."):
        # Invoice
        invoice_text = process_upload(invoice_file)
        invoice_data = map_extracted_text_to_invoice_data_with_confidence_score(invoice_text) if invoice_text else None
        st.subheader("📌 Extracted Fields")
        st.code(invoice_data, language="json")
        st.json(invoice_data)

        # PO
        po_text = robust_extract_text(po_file)
        #po_text = process_upload(po_file)
        st.text_area("Extracted Text from PDF", po_text, height=800)
        po_data = map_extracted_text_to_po_data_with_confidence_score(po_text) if po_text else None  # reuse mapping for PO
        st.subheader("📌 Extracted Fields")
        st.code(po_data, language="json")
        st.json(po_data)

        # Contract
        #contract_text = extract_text_from_docx(contract_file)
        contract_data = extract_contract_fields_with_llm_experimental(contract_file) if contract_file else None
        st.subheader("📌 Extracted Fields")
        st.code(contract_data, language="json")

        # Business Rules
        #rules_text = extract_business_rules_from_docx(rules_file)
        rules_text = robust_extract_text(rules_file)
        rules_data = map_extracted_text_to_business_rules_data_with_confidence_score(rules_text)
        st.subheader("📌 Extracted Fields")
        st.code(rules_data, language="json")
        st.json(rules_data)
        if uploaded_timecard_file_pdf:
            timecard_text = robust_extract_text(uploaded_timecard_file_pdf)
            st.subheader("📋 Extracted Time Card Data")
            st.text_area("Extracted Text from PDF", timecard_text, height=800)
            timecard_data = map_extracted_text_to_timecard_data_with_confidence_score(timecard_text) if timecard_text else None
            st.subheader("📌 Extracted Fields")
            st.code(timecard_data, language="json")
            st.json(timecard_data)
        combined_data = {
            "invoice": invoice_data or {},
            "purchase_order": po_data or {},
            "contract": contract_data  or {},
            "business_rules": rules_data or []
        }

        # SOW processing
        sow_text = extract_business_rules_from_docx(statement_of_work_file) if statement_of_work_file else None
        sow_data = map_extracted_text_to_sow_data_with_confidence_score(sow_text)
        st.subheader("📌 Extracted Fields")
        st.code(sow_data, language="json")
        st.json(sow_data)
        # Prepare agent input
        llm_invoice_agent_input = LLMInvoiceAgentInput(
            input=query,
            input_data=combined_data
        )

        # Call agent
        result = run_llm_invoice_agent(query, combined_data)

        # Output
        st.subheader("🔍 Agent Output")
        st.json(result)

        st.subheader("🧾 Mapped Invoice")
        st.json(invoice_data)

        st.subheader("📦 Mapped Purchase Order")
        st.json(po_data)

        st.subheader("📑 Extracted Contract")
        st.json(contract_data)

        st.subheader("📘 Extracted Business Rules")
        st.json(rules_data)
