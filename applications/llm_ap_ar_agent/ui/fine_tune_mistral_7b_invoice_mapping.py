import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf, safe_json_parse, regex_extract_fields
load_dotenv()

#MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

st.title("📄 Offline Invoice Mapper")

# Upload text input
uploaded_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])
raw_text = ""
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", raw_text, height=200)

# Model settings
    if raw_text and st.button("🧠 Run Mapping"):
        with st.spinner("Running local model..."):
            prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a helpful assistant that extracts structured invoice data from raw text.
             Extract the following fields and return the result strictly as JSON
              (no extra text or explanation):
:
- invoice_id, invoice_date, due_date
- vendor, vendor_address, client, client_address
- line items with description, quantity, unit_price, subtotal
- total_amount, balance_due, service_period
Do not include any explanation. Output only valid JSON.

Raw text:
\"\"\"{text}\"\"\"
    """
    )
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.3,
            max_tokens=512,
            n_ctx=2048,
            verbose=False)
        try:
            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.run({"text": raw_text})
            st.subheader("🧾 Structured Output (from model)")
            st.code(result, language="json")
        except Exception as e:
            print("❌ LLM failed:", str(e))

