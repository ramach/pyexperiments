import re
from datetime import datetime

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_image
from agents.llm_invoice_agent import run_llm_invoice_agent
from dotenv import load_dotenv

st.set_page_config(page_title="LLM Invoice Agent", layout="wide")

st.title("📄💡 LLM-Based Invoice Agent")

load_dotenv()

def extract_fields_from_text(text):
    invoice_id_match = re.search(r"Invoice\s*ID[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    vendor_match = re.search(r"Vendor[:\s]*(.+)", text, re.IGNORECASE)
    amount_match = re.search(r"(?:Total|Amount Due|Amount)[:\s]*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)
    date_match = re.search(r"Date[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)
    due_date_match = re.search(r" Due Date[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)
    po_match = re.search(r"(PO[-][A-Z0-9\-]+)", text, re.IGNORECASE)
    payment_match = re.search(r"Payment\s*Method[:\s]*(.+)", text, re.IGNORECASE)
    return {
        "invoice_id": invoice_id_match.group(1) if invoice_id_match else "AUTO-GEN",
        "vendor": vendor_match.group(1).strip() if vendor_match else "Unknown Vendor",
        "amount": float(amount_match.group(1).replace(",", "")) if amount_match else 0.0,
        "date": date_match.group(1) if date_match else str(datetime.now().date()),
        "due_date": due_date_match.group(1) if due_date_match else str(datetime.now().date()),
        "purchase_order": po_match.group(1) if po_match else "PO-DEFAULT",
        "payment_method": payment_match.group(1).strip().lower().replace(" ", "_") if payment_match else "bank_transfer"
    }

uploaded_file = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
mock_file = st.selectbox("Or select mock invoice data (JSON)", [""] + os.listdir("mockdata/"))

queries = [
    "analyze this invoice",
    "is this invoice valid?",
    "what is the purchase order number?",
    "what is the approval process?",
    "has this invoice been paid?"
]

query_input = st.selectbox("Choose a test query or type your own", queries + ["Custom"])
if query_input == "Custom":
    query = st.text_input("Enter your query")
else:
    query = query_input

submit = st.button("🧠 Run Agent")

if submit:
    with st.spinner("Processing..."):
        invoice_text = ""
        invoice_data = {}

        if uploaded_file:
            file_type = uploaded_file.type
            if "pdf" in file_type:
                invoice_text = extract_text_from_pdf(uploaded_file)
            elif "image" in file_type:
                invoice_text = extract_text_from_image(uploaded_file)
            st.text_area("📄 Extracted Text from PDF", invoice_text, height=200)
        elif mock_file and mock_file != "":
            with open(os.path.join("mockdata", mock_file), "r") as f:
                invoice_data = json.load(f)

        # Run the agent
        if invoice_text or invoice_data:
            guessed_fields = extract_fields_from_text(invoice_text)
            st.markdown("### 🧠 Auto-extracted Fields")
            st.json(guessed_fields)
            if mock_file:
                result = run_llm_invoice_agent(query=query, extracted_text=json.dumps(invoice_data))
            else:
                result = run_llm_invoice_agent(query=query, extracted_text=invoice_text)

            st.write("Results:", result)

