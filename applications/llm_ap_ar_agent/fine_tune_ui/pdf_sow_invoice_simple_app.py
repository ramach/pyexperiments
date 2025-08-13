import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pypdf import PdfReader
import json

# ---------------------
# Config
# ---------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./adapters"  # Path to your LoRA/PEFT adapters

# ---------------------
# Load model + tokenizer
# ---------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        quantization_config=bnb_config
    )
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    except Exception as e:
        st.warning(f"Could not load adapters: {e}")
    return model, tokenizer

# ---------------------
# PDF text extraction
# ---------------------
def read_pdf_text(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# ---------------------
# JSON-safe output parsing
# ---------------------
def try_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------------
# Prompt wrapper
# ---------------------
def build_prompt(text: str, doc_type: str):
    if doc_type.lower() == "invoice":
        instr = """You are a strict JSON extractor for INVOICE documents.
Return ONLY valid minified JSON. Extract:
- title
- invoice_id
- bill_to
- vendor
- date
- amount
- invoice_title
- supplier_information
- period
- terms
- tax
- insurance
- due_date
- payment_method
- additional_text
- line_items
If a field is missing, return "Missing".
"""
    else:
        instr = """You are a strict JSON extractor for STATEMENT OF WORK documents.
Return ONLY valid minified JSON. Extract:
- contracting_company
- vendor
- effective_date
- project_description
- payment_terms
If a field is missing, return "Missing".
"""
    return f"<s>[INST] {instr}\n\nDOCUMENT:\n{text} [/INST]"

# ---------------------
# Generation
# ---------------------
def generate_json(model, tokenizer, prompt: str, max_new_tokens=512):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    data = try_json_load(decoded)
    if data:
        return data, decoded
    return {"error": "Failed to parse JSON."}, decoded

# ---------------------
# Streamlit UI
# ---------------------
st.title("PDF → JSON Field Extractor (TinyLLaMA + Adapters)")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
doc_type = st.selectbox("Document Type", ["Invoice", "Statement of Work"])
if uploaded_file:
    text = read_pdf_text(uploaded_file)
    st.subheader("Extracted Text Preview")
    st.text(text[:1000])

    if st.button("Extract JSON"):
        model, tokenizer = load_model()
        prompt = build_prompt(text, doc_type)
        result, raw = generate_json(model, tokenizer, prompt)
        st.subheader("Extraction Result")
        st.json(result)
        with st.expander("Raw Model Output"):
            st.code(raw)
