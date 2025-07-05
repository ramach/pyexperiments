import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import PyPDF2
import json
import re
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf
load_dotenv()

MODEL_DIR = "./flan_apar_model"

st.set_page_config(page_title="AP/AR Agent with PDF", layout="wide")
st.title("📄 AP/AR LangChain Agent + PDF Upload")

def extract_amount(text):
    # Priority: $amount pattern
    match = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Fallback: number after 'for' or 'amount'
    match = re.search(r"(?:for|amount)\s*(\d+(?:,\d{3})*)", text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))

    # General fallback: any large number
    match = re.search(r"\b(\d{3,})\b", text)
    if match:
        return int(match.group(1).replace(",", ""))

    return None

# Load model + tokenizer
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
    return HuggingFacePipeline(pipeline=hf_pipe)

llm = load_flan_model()
st.success("✅ Fine-tuned FLAN model loaded.")

def run_ap_pipeline(invoice_text: str):
    results = {}
    results["Validation"] = validation_fn(invoice_text)
    results["Business Rule"] = business_rule_fn(invoice_text)
    results["PO Match"] = po_match_fn(invoice_text)
    results["Approval"] = approval_fn(invoice_text)
    results["Payment"] = payment_fn(invoice_text)
    return results

# Tool functions
def validation_fn(invoice_text: str) -> str:
    prompt = f"Validate this invoice: {invoice_text}"
    return llm(prompt)

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

def po_match_fn(invoice_text: str) -> str:
    return "✅ PO matched successfully."

def approval_fn(invoice_text: str) -> str:
    return "✅ Approved by manager."

def payment_fn(invoice_text: str) -> str:
    return "✅ Payment processed."

# Tools + agent
tools = [
    Tool.from_function(validation_fn, name="Validation", description="Validate invoice fields"),
    Tool.from_function(business_rule_fn, name="BusinessRule", description="Check business rule compliance"),
    Tool.from_function(po_match_fn, name="POMatch", description="Match invoice with purchase order"),
    Tool.from_function(approval_fn, name="Approval", description="Approval process check"),
    Tool.from_function(payment_fn, name="Payment", description="Simulate payment processing")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Inputs
st.header("📤 Upload PDF or enter invoice text")
uploaded_pdf = st.file_uploader("Upload invoice PDF", type=["pdf"])

invoice_text = ""
if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        invoice_text = extract_text_from_pdf(uploaded_pdf)
    st.text_area("Extracted text:", invoice_text, height=200)

manual_text = st.text_area("Or manually enter/modify invoice text:", value=invoice_text, height=200)

# Run agent
if st.button("🚀 Run Agent"):
    final_text = manual_text.strip()
    if final_text:
        with st.spinner("Running agent..."):
            results = run_ap_pipeline(final_text)
            st.success("✅ Processing complete")
            st.json(results)
            st.download_button(
                "📥 Download JSON report",
                data=json.dumps(results, indent=2),
                file_name="ap_agent_output.json",
                mime="application/json"
            )
    else:
        st.warning("⚠ Please provide invoice text.")
