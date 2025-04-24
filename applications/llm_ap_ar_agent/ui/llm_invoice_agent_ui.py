import re
from datetime import datetime

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from agents.llm_invoice_agent import run_llm_invoice_agent, map_extracted_text_to_invoice_data_with_confidence_score
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_image
from tools.query_vector_qa import query_vector_similarity_search, seed_faiss_index_if_needed, \
    load_documents_into_vector_store
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

uploaded_invoice = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])
mock_file = st.selectbox("Or select mock invoice data (JSON)", [""] + os.listdir("mockdata/"))

# Upload purchase order PDF or image
uploaded_po = st.file_uploader("Upload Purchase Order (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])


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

if st.button("Run Agent"):
    invoice_text = ""
    po_text = ""

    # Extract from uploaded invoice
    if uploaded_invoice:
        if "pdf" in uploaded_invoice.type:
            invoice_text = extract_text_from_pdf(uploaded_invoice)
        elif "image" in uploaded_invoice.type:
            invoice_text = extract_text_from_image(uploaded_invoice)
            st.text_area("🖼️ Extracted Text from Image", invoice_text, height=200)
            load_documents_into_vector_store([{"text": invoice_text, "source": uploaded_invoice.name}])

    # Extract from uploaded purchase order
    if uploaded_po:
        if "pdf" in uploaded_po.type:
            po_text = extract_text_from_pdf(uploaded_po)
        elif "image" in uploaded_po.type:
            po_text = extract_text_from_image(uploaded_po)

    # Load mock data if selected
    invoice_data = {}
    if mock_file and mock_file != "":
        with open(f"mockdata/{mock_file}", "r") as f:
            invoice_data = json.load(f)

    # Prepare unified input
    input_data = {
        "extracted_text": invoice_text,
        "purchase_order_text": po_text,
        "invoice_json": invoice_data
    }
    combined_text = f"Invoice:\n{invoice_text}\n\nPurchase Order:\n{po_text}"
    mapped_data = map_extracted_text_to_invoice_data_with_confidence_score(combined_text)
    st.subheader("Mapped Invoice + PO data")
    st.json(mapped_data.get("invoice_details", mapped_data))
    # Run the LLM-based agent
    with st.spinner("Running agent..."):
        result = run_llm_invoice_agent(query=query, extracted_text=combined_text)

    # Display results
    st.subheader("Agent Result")
    st.json(result)

