import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf, safe_json_parse, regex_extract_fields
from utils.inference_utils import load_finetuned_model
load_dotenv()
MODEL_DIR = "./models/output_lora_tiny_mapping"

st.set_page_config(page_title="FLAN Invoice Mapper", layout="wide")
st.title("📄 FLAN-based Invoice Field Extractor")

# Load model + tokenizer
@st.cache_resource
def load_flan_model():
    #tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    #model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model,tokenizer = load_finetuned_model(lora_path="./models/output_lora_tiny_mapping")
    hf_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0,
        do_sample=False
    )
    return hf_pipe

hf_pipe = load_flan_model()
st.success("✅ Fine-tuned FLAN model loaded.")

# FLAN mapping prompt
def create_mapping_prompt(extracted_text: str):
    return f"""
    You are an intelligent extraction agent. Extract the values of these fields from the provided invoice text. Respond ONLY with valid JSON containing the extracted values:
- invoice_id
- vendor
- amount
- date
invoice Text:
{extracted_text}

If a field is not present, say "MISSING". invoice_id is same as invoice_number, Respond with JSON only. No additional text.
"""

import json

# Run FLAN + post-process
def run_flan_mapping():
    prompt = create_mapping_prompt(extracted_text)
    output = hf_pipe(prompt)[0]["generated_text"]
    st.code(output, language="json")
    parsed = safe_json_parse(output)
    st.json(parsed)
    try:
        json_str = output.strip()
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        json_str = json_str[start:end]
        return json.loads(json_str)
    except Exception as e:
        output = safe_json_parse(extracted_text)
        return {
            st.json(output)
        }
# Streamlit UI
st.header("📤 Upload Invoice PDF")

uploaded_invoice = st.file_uploader("Upload Invoice PDF", type=["pdf"])
extracted_text = ""

if uploaded_invoice:
    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_pdf(uploaded_invoice)
    st.text_area("Extracted Invoice Text", extracted_text, height=200)

if st.button("🚀 Extract Fields via FLAN"):
    if extracted_text.strip():
        with st.spinner("Running FLAN mapping..."):
            result = run_flan_mapping()
        st.success("✅ Mapping complete")
        st.json(result)
        st.download_button(
            "📥 Download mapped data",
            data=json.dumps(result, indent=2),
            file_name="invoice_mapped.json",
            mime="application/json"
        )
    else:
        st.warning("⚠ Please provide invoice text")
