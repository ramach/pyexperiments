## streamlit_ui/invoice_agent_ui.py
import streamlit as st
import sys
import os
#Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# streamlit_ui/invoice_agent_ui.py
import sqlite3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from agents.invoice_agent import run_invoice_agent_with_function_calling
from tools.openai_functions.ap_tools import run_ap_chain_with_function_calling
from tools.openai_functions.ar_tools import run_ar_chain_with_function_calling
from retrievers.serpapi_scraper import run_serpapi_scraper
from tools.query_vector_qa import query_vector_similarity_search, seed_faiss_index_if_needed, load_documents_into_vector_store
import json
from datetime import datetime

import re

load_dotenv()

st.set_page_config(page_title="Invoice Agent", layout="wide")
st.title("📄 Invoice Agent with Multi-Agent Routing")

# Sidebar for agent selection
agent_type = st.sidebar.selectbox("Select Agent", [
    "Invoice", "Accounts Payable", "Accounts Receivable", "Similarity Search"
])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# SQLite database setup
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

# PDF Upload
uploaded_file = st.file_uploader("Upload Invoice PDF (optional)", type=["pdf"])
extracted_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"
    st.text_area("Extracted Text from PDF", extracted_text, height=200)

    # Load into vector store
    with st.spinner("Ingesting uploaded PDF into vector store..."):
        load_documents_into_vector_store([{"text": extracted_text, "source": uploaded_file.name}])

# Helper to extract fields if form is not filled

def extract_fields_from_text(text):
    invoice_id_match = re.search(r"Invoice\s*ID[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    vendor_match = re.search(r"Vendor[:\s]*(.+)", text, re.IGNORECASE)
    amount_match = re.search(r"(?:Total|Amount Due|Amount)[:\s]*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)
    date_match = re.search(r"Date[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)
    po_match = re.search(r"PO(?:\s*Number)?[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    payment_match = re.search(r"Payment\s*Method[:\s]*(.+)", text, re.IGNORECASE)

    return {
        "invoice_id": invoice_id_match.group(1) if invoice_id_match else "AUTO-GEN",
        "vendor": vendor_match.group(1).strip() if vendor_match else "Unknown Vendor",
        "amount": float(amount_match.group(1).replace(",", "")) if amount_match else 0.0,
        "date": date_match.group(1) if date_match else str(datetime.now().date()),
        "due_date": str(datetime.now().date()),
        "purchase_order": po_match.group(1) if po_match else "PO-DEFAULT",
        "payment_method": payment_match.group(1).strip().lower().replace(" ", "_") if payment_match else "bank_transfer"
    }

if uploaded_file and extracted_text:
    guessed_fields = extract_fields_from_text(extracted_text)
    st.markdown("### 🧠 Auto-extracted Fields (editable in form):")
    st.json(guessed_fields)

# Invoice Form
with st.form("invoice_form"):
    st.subheader("Invoice Details")
    invoice_id = st.text_input("Invoice ID")
    vendor = st.text_input("Vendor")
    amount = st.number_input("Amount", min_value=0.0)
    date = st.date_input("Date")
    purchase_order = st.text_input("Purchase Order")
    payment_method = st.selectbox("Payment Method", ["bank_transfer", "credit_card", "check"])
    question = st.text_input("Ask a question (for similarity search)")

    submitted = st.form_submit_button("Submit")

# Ensure FAISS index is seeded
seed_faiss_index_if_needed()

# Main Agent Execution
if submitted:
    extracted_fields = extract_fields_from_text(extracted_text)

    payload = {
        "invoice_id": invoice_id or extracted_fields["invoice_id"],
        "vendor": vendor or extracted_fields["vendor"],
        "amount": amount or extracted_fields["amount"],
        "date": str(date) if date else extracted_fields["date"],
        "due_date": str(date) if date else extracted_fields["date"],
        "purchase_order": purchase_order or extracted_fields["purchase_order"],
        "payment_method": payment_method or extracted_fields["payment_method"],
        "extracted_text": extracted_text
    }
    st.markdown("### 🔍 JSON Payload to Agent:")
    st.code(json.dumps(payload, indent=2))
    json_payload = json.dumps(payload)

    with st.spinner("Running selected agent..."):
        try:
            if agent_type == "Invoice":
                response = run_invoice_agent_with_function_calling(json_payload)
            elif agent_type == "Accounts Payable":
                response = run_ap_chain_with_function_calling(json_payload)
            elif agent_type == "Accounts Receivable":
                response = run_ar_chain_with_function_calling(json_payload)
            elif agent_type == "Similarity Search":
                if question:
                    response = query_vector_similarity_search(question)
                else:
                    response = "Please enter a question for similarity search."
            else:
                response = "Invalid agent type."

            public_data = run_serpapi_scraper(payload["vendor"])

            st.success("Agent Response")
            st.write(response)
            st.subheader("Vendor Public Info")
            st.write(public_data)

            st.session_state.history.append({"agent": agent_type, "input": payload, "response": response})

            c.execute("""
                INSERT INTO invoice_history (timestamp, agent_type, payload, response)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), agent_type, json.dumps(payload), response))
            conn.commit()

        except Exception as e:
            st.error(f"[Error] {str(e)}")

# Chat History Display
with st.sidebar:
    st.subheader("🕘 Chat History")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**{item['agent']}**: {json.dumps(item['input'])}")
        st.markdown(f"_Response_: {item['response'][:100]}...")
