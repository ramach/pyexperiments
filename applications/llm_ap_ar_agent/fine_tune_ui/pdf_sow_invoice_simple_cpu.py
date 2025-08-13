
import re
import json
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- PDF reading ----
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader  # fallback

st.set_page_config(page_title="PDF → JSON Extractor (CPU/MPS, no bitsandbytes)", layout="wide")
st.title("📄 PDF → JSON Extractor (TinyLLaMA/Mistral + LoRA) — CPU/MPS Safe")

with st.sidebar:
    st.header("Model Settings")
    base_model_id = st.text_input("Base model (folder or HF id)", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_path  = st.text_input("LoRA adapter folder", value="./models/tinyllama_trained_adapter_mapping_2")
    device_choice = st.selectbox("Device", ["auto (prefer MPS)", "cpu", "mps"], index=0)
    max_new_tokens = st.slider("max_new_tokens", 64, 1536, 384, step=32)

# ---- device resolve (no bitsandbytes) ----
def resolve_device(choice: str):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # auto
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

DEVICE = resolve_device(device_choice)

# ---- helpers ----
def read_pdf_text(file) -> str:
    reader = PdfReader(file)
    chunks = []
    for p in getattr(reader, "pages", []):
        try:
            chunks.append(p.extract_text() or "")
        except Exception:
            pass
    return "\n".join(chunks)

def clean_pdf_text(raw: str, limit=16000) -> str:
    # Avoid accidental instruction tokens present in docs
    raw = raw.replace("[INST]", "[INST_]").replace("[/INST]", "[/_INST]")
    # Basic cleanup
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()[:limit]

def extract_first_json_block(s: str):
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = s[start:end+1]
        try:
            return json.loads(blob)
        except Exception:
            pass
    # regex fallback (nested braces)
    m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # last-ditch: try to fix truncated braces
    m2 = re.search(r"\{.*", s, flags=re.S)
    if m2:
        blob = m2.group(0)
        opens, closes = blob.count("{"), blob.count("}")
        if opens > closes:
            blob += "}" * (opens - closes)
        try:
            return json.loads(blob)
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(base_model_id: str, adapter_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,              # CPU/MPS friendly
        low_cpu_mem_usage=True,
        device_map=None,                        # we'll move manually
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return tok, model

def build_invoice_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for INVOICE documents.\n"
        "Return ONLY one valid minified JSON object. No markdown, no explanations.\n"
        "Fields: title, invoice_id, bill_to, vendor, date, amount, invoice_title, "
        "supplier_information, period, terms, tax, insurance, due_date, payment_method, "
        "additional_text, line_items.\n"
        "Rules:\n"
        "- If a field is not present, use \"Missing\".\n"
        "- If invoice_title is missing, set it to \"Invoice\".\n"
        "- Use \"Remit To\" as vendor if available; if vendor name missing, use business_id or client_id.\n"
        "- Split vendor and bill_to into name, address, phone, email if possible.\n"
        "- Use additional_text for anything not covered.\n"
        "Output MUST start with '{' and end with '}'."
    )
    return f"{instr}\n\nDOCUMENT:\n{doc_text}"

def build_sow_prompt(mainline_text: str) -> str:
    instr = (
        "Extract contracting company and vendor from the SOW main sentence.\n"
        "Return ONLY a minified JSON object with exactly: {\"contracting_company\":..., \"vendor\":...}.\n"
        "The vendor MUST appear in the main line containing the word 'between'.\n"
        "If not present there, set vendor to \"vendor as legal contract is missing\"."
    )
    return f"{instr}\n\nSENTENCE:\n{mainline_text}"

def wrap_as_inst(s: str) -> str:
    # Match LoRA training style
    return f"<s>[INST] {s.strip()} [/INST]"

def generate_completion_json(model, tokenizer, prompt: str, device: torch.device, max_new_tokens=384):
    # encode
    enc = tokenizer(wrap_as_inst(prompt), return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc.input_ids.to(device)
    input_len = input_ids.shape[-1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.02,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    tail_ids = out[0][input_len:]  # completion only
    completion = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()

    js = extract_first_json_block(completion)
    if js is not None:
        return js, completion
    return {"error": "Failed to parse JSON from model output."}, completion

# ---- UI ----
tab1, tab2 = st.tabs(["SOW (main line only)", "Invoice (compact schema)"])

with tab1:
    st.subheader("SOW → contracting_company & vendor from main line")
    sow_pdf = st.file_uploader("Upload SOW PDF", type=["pdf"], key="sow_pdf")
    if sow_pdf:
        with st.spinner("Reading PDF…"):
            raw = read_pdf_text(sow_pdf)
            text = clean_pdf_text(raw)
            # try to find the main line
            flat = " ".join(text.split())
            m = re.search(r"(Statement of Work.*?between.*?\.)", flat, flags=re.I)
            mainline = m.group(1).strip() if m else text[:300]

        st.text_area("Detected main line", value=mainline, height=120)

        if st.button("Extract SOW JSON"):
            tok, model = load_model_and_tokenizer(base_model_id, adapter_path, DEVICE)
            prompt = build_sow_prompt(mainline)
            js, raw_out = generate_completion_json(model, tok, prompt, DEVICE, max_new_tokens=max_new_tokens)
            st.subheader("Parsed JSON")
            st.json(js)
            with st.expander("Raw model completion"):
                st.code(raw_out)

with tab2:
    st.subheader("Invoice → minimal fields")
    inv_pdf = st.file_uploader("Upload Invoice PDF", type=["pdf"], key="inv_pdf")
    if inv_pdf:
        with st.spinner("Reading PDF…"):
            raw = read_pdf_text(inv_pdf)
            doc = clean_pdf_text(raw)

        st.text_area("Document preview (cleaned)", value=doc[:1200], height=160)

        if st.button("Extract Invoice JSON"):
            tok, model = load_model_and_tokenizer(base_model_id, adapter_path, DEVICE)
            prompt = build_invoice_prompt(doc)
            js, raw_out = generate_completion_json(model, tok, prompt, DEVICE, max_new_tokens=max_new_tokens)
            st.subheader("Parsed JSON")
            st.json(js)
            with st.expander("Raw model completion"):
                st.code(raw_out)
