import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
import re
import yaml
import pdfplumber
from pdf2image import convert_from_path
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from agents.invoice_agent import run_invoice_agent_with_function_calling
from tools.openai_functions.ap_tools import run_ap_chain_with_function_calling
from tools.openai_functions.ar_tools import run_ar_chain_with_function_calling
from retrievers.serpapi_scraper import run_serpapi_scraper
from tools.query_vector_qa import query_vector_similarity_search, seed_faiss_index_if_needed, \
    load_documents_into_vector_store
from utils.pdf_utils import extract_text_from_pdf
from utils.pdf_utils import extract_text_from_image

load_dotenv()

def extract_text_from_pdf(file_path):
    extracted_text = ""

    # Try with pdfplumber first (good for structured text and tables)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""
    except Exception as e:
        print(f"[PDFPlumber Error] {e}")

    # If nothing was extracted or very little text, fallback to OCR
    if len(extracted_text.strip()) < 50:
        print("[OCR Fallback] Using OCR for image-based PDF")
        images = convert_from_path(file_path)
        for img in images:
            extracted_text += pytesseract.image_to_string(img)

    return extracted_text.strip()


def load_mock_data(filename):
    filepath = os.path.join("mockdata", filename)
    with open(filepath, "r") as f:
        return json.load(f)


st.set_page_config(page_title="Invoice Agent", layout="wide")
st.title("📄 Invoice Agent with Multi-Agent Routing")

agent_type = st.sidebar.selectbox("Select Agent", [
    "Invoice", "Accounts Payable", "Accounts Receivable", "Similarity Search"
])

# Session state init
if "history" not in st.session_state:
    st.session_state.history = []

# SQLite DB setup
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

# PDF or Image upload
uploaded_pdf = st.file_uploader("Upload Invoice PDF", type=["pdf"])
uploaded_image = st.file_uploader("Upload Invoice Image", type=["png", "jpg", "jpeg"])
mock_data_file = st.file_uploader("Upload Mock JSON", type=["json"])

extracted_text = ""

# Handle PDF
if uploaded_pdf:
    extracted_text = extract_text_from_pdf(uploaded_pdf)
    '''
    reader = PdfReader(uploaded_pdf)
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"
        '''

    st.text_area("📄 Extracted Text from PDF", extracted_text, height=200)
    load_documents_into_vector_store([{"text": extracted_text, "source": uploaded_pdf.name}])

# Handle image
elif uploaded_image:
    image = Image.open(uploaded_image)
    extracted_text = pytesseract.image_to_string(image)
    st.text_area("🖼️ Extracted Text from Image", extracted_text, height=200)
    load_documents_into_vector_store([{"text": extracted_text, "source": uploaded_image.name}])

# Handle mock JSON
elif mock_data_file:
    try:
        loaded = json.load(mock_data_file)
        #loaded = load_mock_data(mock_data_file)
        mock_choice = st.selectbox("Select mock invoice:",
                                   options=[f"{i + 1}: {d['invoice_id']}" for i, d in enumerate(loaded)])
        selected_data = loaded[int(mock_choice.split(":")[0]) - 1]
        st.json(selected_data)
        # Support both single dict and list of dicts
        if isinstance(loaded, list) and len(loaded) > 0:
            mock_data = loaded[0]
        elif isinstance(loaded, dict):
            mock_data = loaded
        else:
            raise ValueError("Invalid mock JSON format")

        # Extract values with fallback defaults
        invoice_id = mock_data.get('invoice_id', 'UNKNOWN')
        vendor = mock_data.get('vendor', 'UNKNOWN')
        date = mock_data.get('date', 'UNKNOWN')
        amount = mock_data.get('amount', '0.00')
        po = mock_data.get('purchase_order', 'UNKNOWN')
        payment = mock_data.get('payment_method', 'UNKNOWN')
        extracted_text = (
            f"Invoice ID: {invoice_id}\n"
            f"Vendor: {vendor}\n"
            f"Date: {date}\n"
            f"Amount: ${amount}\n"
            f"Purchase Order: {po}\n"
            f"Payment Method: {payment}\n"
        )

        st.success("Mock JSON loaded.")
        st.text_area("Extracted from Mock JSON", extracted_text, height=200)

    except Exception as e:
        st.error(f"Failed to parse mock file: {str(e)}")


# Field extractor
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


if extracted_text:
    guessed_fields = extract_fields_from_text(extracted_text)
    st.markdown("### 🧠 Auto-extracted Fields")
    st.json(guessed_fields)

# Invoice form
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

# Seed FAISS
seed_faiss_index_if_needed()

# Main execution
if submitted:
    extracted_fields = extract_fields_from_text(extracted_text)
    payload = {
        "invoice_id": invoice_id or extracted_fields["invoice_id"],
        "vendor": vendor or extracted_fields["vendor"],
        "amount": amount or extracted_fields["amount"],
        "date": str(date) if date else extracted_fields["date"],
        "due_date": extracted_fields["due_date"],
        "purchase_order": purchase_order or extracted_fields["purchase_order"],
        "payment_method": payment_method or extracted_fields["payment_method"],
        "extracted_text": extracted_text
    }

    try:
        with st.spinner("Running agent..."):
            payload_str = json.dumps(payload)

            if agent_type == "Invoice":
                response = run_invoice_agent_with_function_calling(payload_str)
            elif agent_type == "Accounts Payable":
                response = run_ap_chain_with_function_calling(payload_str)
            elif agent_type == "Accounts Receivable":
                response = run_ar_chain_with_function_calling(payload_str)
            elif agent_type == "Similarity Search":
                response = query_vector_similarity_search(question) if question else "Please enter a question."
            else:
                response = "Invalid agent type"

            public_data = run_serpapi_scraper(payload["vendor"])

            st.success("Agent Response")
            st.write(response)

            st.subheader("🔍 Vendor Info (via SerpAPI)")
            st.write(public_data)

            st.session_state.history.append({"agent": agent_type, "input": payload, "response": response})

            c.execute("""
                INSERT INTO invoice_history (timestamp, agent_type, payload, response)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), agent_type, json.dumps(payload), response))
            conn.commit()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Chat History
with st.sidebar:
    st.subheader("🕘 Chat History")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"**{item['agent']}**: {json.dumps(item['input'])}")
        st.markdown(f"_Response_: {item['response'][:100]}...")
