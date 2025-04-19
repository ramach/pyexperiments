# streamlit_ui/invoice_agent_ui.py

import streamlit as st
import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from agents.invoice_agent import run_invoice_agent_with_function_calling
from tools.query_vector_qa import seed_faiss_index_if_needed, load_documents_into_vector_store
from retrievers.serpapi_scraper import run_serpapi_scraper

import re

from llm_ap_ar_agent.ui.invoice_agent_ui import extract_fields_from_text

load_dotenv()
st.set_page_config(page_title="Invoice Agent", layout="wide")
st.title("📄 Invoice Agent (Auto Run All Tools)")

agent_type = "Invoice"

if "history" not in st.session_state:
    st.session_state.history = []

# SQLite setup
DB_PATH = "invoice_agent.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS invoice_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        agent_type TEXT,
        payload TEXT,
        response TEXT
    )
''')
conn.commit()
# PDF or image upload
uploaded_file = st.file_uploader("Upload Invoice PDF/Image (optional)", type=["pdf", "png", "jpg", "jpeg"])
extracted_text = ""

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
    else:
        import pytesseract
        from PIL import Image
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image)

    st.text_area("Extracted Text", extracted_text, height=200)
    load_documents_into_vector_store([{"text": extracted_text, "source": uploaded_file.name}])

# Mock JSON data upload
mock_file = st.file_uploader("Or Upload Mock Invoice JSON", type=["json"])
guessed_fields = {}

if mock_file:
    try:
        mock_data = json.load(mock_file)
        invoice_id = mock_data.get("invoice_id", "UNKNOWN")
        vendor = mock_data.get("vendor", "UNKNOWN")
        date = mock_data.get("date", "UNKNOWN")
        amount = mock_data.get("amount", "0.00")
        po = mock_data.get("purchase_order", "UNKNOWN")
        payment = mock_data.get("payment_method", "UNKNOWN")

        extracted_text = f"""Invoice ID: {invoice_id}
Vendor: {vendor}
Date: {date}
Amount: ${amount}
Purchase Order: {po}
Payment Method: {payment}
"""
        guessed_fields = mock_data
        st.text_area("Extracted (from mock)", extracted_text, height=200)
    except Exception as e:
        st.error(f"[Error parsing mock JSON] {e}")
# Field extraction if available
if not guessed_fields and extracted_text:
    guessed_fields = extract_fields_from_text(extracted_text)
    st.markdown("### 🧠 Auto-extracted Fields (editable in form):")
    st.json(guessed_fields)

# Invoice Form
with st.form("invoice_form"):
    st.subheader("Invoice Details")
    invoice_id = st.text_input("Invoice ID", guessed_fields.get("invoice_id", ""))
    vendor = st.text_input("Vendor", guessed_fields.get("vendor", ""))
    amount = st.number_input("Amount", min_value=0.0, value=guessed_fields.get("amount", 0.0))
    date = st.date_input("Date", datetime.fromisoformat(guessed_fields.get("date", str(datetime.now().date()))))
    purchase_order = st.text_input("Purchase Order", guessed_fields.get("purchase_order", ""))
    payment_method = st.selectbox("Payment Method", ["bank_transfer", "credit_card", "check"], index=0)
    submitted = st.form_submit_button("Submit")

# Ensure FAISS index is seeded
seed_faiss_index_if_needed()
# Main Agent Execution
# PART 4 - Main Agent Execution
# PART 4 - Main Agent Execution
if submitted:
    # Extract fields from text as fallback
    extracted_fields = extract_fields_from_text(extracted_text)

    payload = {
        "invoice_id": invoice_id or extracted_fields["invoice_id"],
        "vendor": vendor or extracted_fields["vendor"],
        "amount": amount or extracted_fields["amount"],
        "date": str(date) if date else extracted_fields["date"],
        "purchase_order": purchase_order or extracted_fields["purchase_order"],
        "payment_method": payment_method or extracted_fields["payment_method"],
        "extracted_text": extracted_text
    }

    query = f"""Verify invoice from {payload['vendor']} dated {payload['date']} with amount ${payload['amount']}"""

    st.write(f"Running query: `{query}`")

    try:
        with st.spinner("Running invoice agent..."):
            result = agent.run(query)

        st.subheader("🔍 Invoice Verification Result")
        st.write(result.get("invoice_verification", "No result returned."))

        st.subheader("📦 PO Matching Result")
        st.write(result.get("po_matching", "No result returned."))

        st.subheader("✅ Approval Process Result")
        st.write(result.get("approval_process", "No result returned."))

        st.subheader("💰 Payment Processing Result")
        st.write(result.get("payment_processing", "No result returned."))

        st.session_state.history.append({"input": payload, "response": result})

    except Exception as e:
        st.warning(f"[Agent fallback] Agent failed with: {str(e)}. Executing tools manually...")

        try:
            # Manual tool execution fallback
            result = {
                "invoice_verification": run_invoice_verification(json.dumps(payload)),
                "po_matching": run_po_matching(json.dumps(payload)),
                "approval_process": run_approval_process(json.dumps(payload)),
                "payment_processing": run_payment_processing(json.dumps(payload)),
            }

            st.subheader("🔍 Invoice Verification Result")
            st.write(result["invoice_verification"])

            st.subheader("📦 PO Matching Result")
            st.write(result["po_matching"])

            st.subheader("✅ Approval Process Result")
            st.write(result["approval_process"])

            st.subheader("💰 Payment Processing Result")
            st.write(result["payment_processing"])

            st.session_state.history.append({"input": payload, "response": result})

        except Exception as inner_e:
            st.error(f"[Fallback Error] {str(inner_e)}")

# Sidebar History
with st.sidebar:
    st.subheader("🕘 Chat History")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**{item['agent']}**: {json.dumps(item['input'])}")
        st.markdown(f"_Verification_: {item['response']['verification'][:100]}...")
        st.markdown(f"_PO Match_: {item['response']['po_matching'][:100]}...")