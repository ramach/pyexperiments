
import streamlit as st
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf
load_dotenv()

from utils.field_extractor import extract_invoice_fields
import fitz  # PyMuPDF

st.set_page_config(page_title="Invoice Field Extractor", layout="wide")

st.title("🧾 Invoice Field Extractor (Local + Hybrid)")
st.write("Upload an invoice PDF or paste text, and extract key fields using local models with optional GPT fallback.")

upload_option = st.radio("Input Type", ["Upload PDF", "Paste Text"])

invoice_text = ""
if upload_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_file:
        invoice_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF loaded successfully.")
elif upload_option == "Paste Text":
    invoice_text = st.text_area("Paste raw invoice text here", height=300)

fallback = st.checkbox("Use GPT fallback if local extraction fails", value=True)

if invoice_text and st.button("Extract Fields"):
    with st.spinner("Extracting..."):
        try:
            result = extract_invoice_fields(invoice_text, fallback_to_gpt=fallback)
            st.subheader("📦 Extracted Fields")
            st.json(result)
        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
