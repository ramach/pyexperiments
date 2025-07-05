import json

import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from utils.inference_utils import load_finetuned_model, run_invoice_toolchain
from utils.toolchain_utils import *
from utils.field_extractor import extract_invoice_fields

st.set_page_config(page_title="TinyLlama Invoice Toolchain", layout="wide")
st.title("🧾 TinyLlama-Powered Invoice Toolchain")

@st.cache_resource
def load_model():
    return load_finetuned_model()

model, tokenizer = load_model()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

st.sidebar.header("📎 Upload Invoice")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

st.sidebar.header("🧠 Task Selector")
selected_task = st.sidebar.selectbox("Choose a task", [
    "Invoice Field Verification",
    "PO Matching",
    "Approval Decision",
    "Payment Eligibility"
])

if uploaded_file:
    filetype = uploaded_file.type
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "pdf" in filetype:
        invoice_text = extract_text_from_pdf(tmp_path)
    elif "image" in filetype:
        invoice_text = extract_text_from_image(tmp_path)
    else:
        invoice_text = ""

    os.unlink(tmp_path)

    if invoice_text.strip():
        st.subheader("📄 Extracted Invoice Text")
        st.code(invoice_text.strip()[:2000], language="markdown")
        mapped_data = extract_invoice_fields(invoice_text, True)
        #query = f"Extract invoice fields as JSON:\n{invoice_text}"
        structured_invoice_data = json.dumps(mapped_data, indent=2)
        query = f"Extract invoice fields as JSON:\n{structured_invoice_data}"
        with st.spinner("🔍 Extracting fields..."):
            response = run_invoice_toolchain(query, model, tokenizer)
            parsed_fields = parse_llm_output_to_dict(response)

        st.subheader("📦 Extracted Fields")
        st.json(parsed_fields)

        # Run specific toolchain function
        st.subheader("✅ Toolchain Result")
        if selected_task == "Invoice Field Verification":
            result = verify_invoice_fields(parsed_fields)
        elif selected_task == "PO Matching":
            result = run_po_matching(parsed_fields)
        elif selected_task == "Approval Decision":
            result = check_approval_required(parsed_fields)
        elif selected_task == "Payment Eligibility":
            result = check_payment_eligibility(parsed_fields)
        else:
            result = {"error": "Unknown task"}

        st.json(result)
    else:
        st.warning("❗ No extractable text found.")
else:
    st.info("📂 Upload a PDF or image invoice to begin.")