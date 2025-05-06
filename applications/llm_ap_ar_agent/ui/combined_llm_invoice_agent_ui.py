import streamlit as st
import json
import os
import sys
from io import BytesIO

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf
from utils.text_extraction import extract_text_from_image
from utils.text_extraction import extract_business_rules_from_docx
from agents.llm_business_rule_agent import map_rule_text_to_structured
from agents.llm_invoice_agent import map_extracted_text_to_invoice_data_with_confidence_score, map_extracted_text_to_invoice_data
from agents.combined_invoice_processing_with_business_rule import run_llm_invoice_agent

load_dotenv()

st.title("LLM Invoice Agent with Combined Data")

st.sidebar.header("Upload Files")
invoice_file = st.sidebar.file_uploader("Invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
po_file = st.sidebar.file_uploader("Purchase Order (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
rules_file = st.sidebar.file_uploader("Business Rules (DOCX)", type=["docx"])

query = st.selectbox("Choose a query", [
    "Analyze this invoice",
    "Is this invoice valid?",
    "What is the approval process?",
    "How should we process the payment?"
])

if st.button("Run Invoice Agent"):
    invoice_text, po_text, mapped_data, mapped_data_without_score, rules_data = {}, {}, {}, {}, {}

    # 1. Extract + Map Invoice
    if invoice_file:
        if "pdf" in invoice_file.type:
            invoice_text = extract_text_from_pdf(invoice_file)
        else:
            invoice_text = extract_text_from_image(invoice_file)

    # 2. Extract + Map PO
    if po_file:
        if "pdf" in po_file.type:
            po_text = extract_text_from_pdf(po_file)
        else:
            po_text = extract_text_from_image(po_file)
    if invoice_text and po_text:
        combined_text = invoice_text + "\n\n" + po_text
        mapped_data = map_extracted_text_to_invoice_data_with_confidence_score(combined_text)
        mapped_data_without_score = map_extracted_text_to_invoice_data(combined_text)
    st.subheader("Mapped Invoice + PO data")
    st.json(mapped_data.get("invoice_details", mapped_data))
    # 3. Extract Business Rules
    rules = []
    if rules_file:
        #get raw rules as list
        rules = extract_business_rules_from_docx(rules_file)
    if not rules:
        st.warning("No rules detected in the document.")
    else:
        st.success(f"✅ Found {len(rules)} candidate rules")
        mapped_rules = []
        for i, rule in enumerate(rules):
            with st.expander(f"Rule {i+1} (Raw Text)"):
                st.text(rule)

                with st.spinner("Mapping rule with LLM..."):
                    mapped = map_rule_text_to_structured(rule)
                    st.json(mapped)
                    mapped_rules.append(mapped)

        if mapped_rules:
            json_output = json.dumps(mapped_rules, indent=2)
            json_bytes = BytesIO(json_output.encode("utf-8"))
            st.download_button("📥 Download Mapped Rules (JSON)", data=json_bytes, file_name="mapped_business_rules.json", mime="application/json")
        # 4. Combine rules and invoice_PO
        combined_input = {
            "invoice_and_PO_details_with_confidence": mapped_data,
            "business_rules": mapped_rules
        }
        # 5. Show Combined Data
        st.subheader("Combined Mapped Input")
        st.json(combined_input)
        # remove confidence score while passing to agent
        combined_input_without_score = {
            "invoice_and_PO_details": mapped_data_without_score,
            "business_rules": mapped_rules
        }
        try:
            results = run_llm_invoice_agent(query=query, input_data=combined_input_without_score)
            st.subheader("Agent Responses")
            for tool_name, result in results.items():
                st.markdown(f"**{tool_name.replace('_', ' ').title()}**: {result}")
        except Exception as e:
            st.error(f"Agent failed: {e}")

