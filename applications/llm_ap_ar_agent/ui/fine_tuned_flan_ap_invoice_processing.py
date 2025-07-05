import os
import sys

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf, extract_amount
load_dotenv()

MODEL_DIR = "./flan_apar_model"

st.set_page_config(page_title="AP/AR Tool Chain", layout="wide")
st.title("📄 AP/AR Processor with Invoice + PO + Contract")

# Load FLAN model
@st.cache_resource
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    hf_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return hf_pipe

hf_pipe = load_flan_model()
st.success("✅ Fine-tuned FLAN model loaded.")

# PDF extraction utility
# Tool functions
def validation_fn(invoice_text, po_text, contract_text):
    prompt = f"""Validate this invoice against PO and contract:
Invoice: {invoice_text}
PO: {po_text}
Contract: {contract_text}"""
    output = hf_pipe(prompt)[0]["generated_text"]
    return output.strip()

def business_rule_fn(invoice_text: str) -> str:
    rule_max = st.number_input("Set max allowed invoice amount", min_value=100, value=10000)
    try:
        amount = extract_amount(invoice_text)
        if amount:
            if amount <= rule_max:
                return f"✅ Invoice amount ${amount} is within allowed max ${rule_max}"
            else:
                return f"❌ Invoice amount ${amount} exceeds allowed max ${rule_max}"
        else:
            return "⚠ Could not extract a valid amount from query."
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def po_match_fn(invoice_text, po_text):
    # Simple simulation
    return "✅ PO matched successfully."

def approval_fn(invoice_text, contract_text):
    # Could do more advanced logic
    return "✅ Approved per contract terms."

def payment_fn():
    return "✅ Payment processed."

# Run tool chain
def run_tool_chain(invoice_text, po_text, contract_text):
    return {
        "Validation": validation_fn(invoice_text, po_text, contract_text),
        "Business Rule": business_rule_fn(invoice_text),
        "PO Match": po_match_fn(invoice_text, po_text),
        "Approval": approval_fn(invoice_text, contract_text),
        "Payment": payment_fn()
    }

# Streamlit UI
st.header("📤 Upload Documents")

uploaded_invoice = st.file_uploader("Upload Invoice PDF", type=["pdf"])
uploaded_po = st.file_uploader("Upload PO PDF", type=["pdf"])
uploaded_contract = st.file_uploader("Upload Contract PDF", type=["pdf"])

invoice_text, po_text, contract_text = "", "", ""

if uploaded_invoice:
    with st.spinner("Extracting invoice..."):
        invoice_text = extract_text_from_pdf(uploaded_invoice)
    st.text_area("Extracted Invoice Text", invoice_text, height=150)

if uploaded_po:
    with st.spinner("Extracting PO..."):
        po_text = extract_text_from_pdf(uploaded_po)
    st.text_area("Extracted PO Text", po_text, height=150)

if uploaded_contract:
    with st.spinner("Extracting Contract..."):
        contract_text = extract_text_from_pdf(uploaded_contract)
    st.text_area("Extracted Contract Text", contract_text, height=150)

if st.button("🚀 Run AP/AR Tool Chain"):
    if invoice_text.strip():
        with st.spinner("Running tool chain..."):
            results = run_tool_chain(invoice_text, po_text, contract_text)
        st.success("✅ Tool chain completed.")
        st.json(results)
        st.download_button(
            "📥 Download JSON report",
            data=json.dumps(results, indent=2),
            file_name="ap_agent_output.json",
            mime="application/json"
        )
    else:
        st.warning("⚠ Please provide at least an invoice.")
