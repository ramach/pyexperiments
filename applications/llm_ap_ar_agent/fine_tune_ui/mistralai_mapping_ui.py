import os, re, json, gc, time
from typing import Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import torch
from pypdf import PdfReader
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CPU-friendly defaults
# =========================
# Notes:
# - 7B on CPU still needs a *lot* of RAM (12–16 GB+). Close other apps.
# - We avoid bitsandbytes here (GPU-only). This is pure CPU inference.
# - Set conservative generation params by default.

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_NEW = 384              # keep smaller for CPU memory/time
DEFAULT_MAX_DOC_CHARS = 16000      # trim long PDFs so input stays within context
DEFAULT_NUM_THREADS = max(os.cpu_count() - 1, 1)

# If you run on Apple Silicon, uncomment to prefer mps (still experimental).
# USE_MPS = torch.backends.mps.is_available()
USE_MPS = False

# =========================
# Utility
# =========================
def wrap_inst(s: str) -> str:
    return f"<s>[INST] {s.strip()} [/INST]"

def read_pdf_text(path: str, max_chars: int) -> str:
    r = PdfReader(path)
    parts = []
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ""
            parts.append(f"\n--- Page {i+1} ---\n{t}")
        except Exception:
            continue
    text = "\n".join(parts)
    # sanitize chat markers
    text = text.replace("[/INST]", " ").replace("[INST]", " ")
    return text[:max_chars]

def build_invoice_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for INVOICE documents.\n"
        "Return ONLY one minified JSON object between <json> and </json>. No markdown, no explanations.\n"
        "Fields: title, invoice_id, bill_to, vendor, date, amount, "
        "supplier_information, period, terms, tax, insurance, due_date, payment_method, "
        "additional_text, line_items.\n"
        "Rules:\n"
        "- If a field is not present, set it to \"Missing\".\n"
        "- If invoice_title or title is missing, set it to \"Invoice\".\n"
        "- line_items can have more than one line. QTY or qty or quantity is quantity; unit price may be written as 'unit price' or 'unitprice'.\n"
        "- 'additional notes' may map to additional_text.\n"
        "- If 'Remit To' exists, treat it as vendor; otherwise vendor may be business_id/client_id header.\n"
        "- Split vendor and bill_to into name, address, phone, email if possible.\n"
        "- Put any extra content under additional_text.\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

def build_sow_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for STATEMENT OF WORK (SOW) documents.\n"
        "Return ONLY one minified JSON object between <json> and </json>. No markdown, no explanations.\n"
        "Fields: title, sow_number, sow_date, agreement_date, parties, services_summary, term, roles, "
        "fees, billing, expenses, termination, contacts, signatures, additional_text.\n"
        "Rules:\n"
        "- If a field is missing, set it to \"Missing\" (or [] / {} if it is a list/object).\n"
        "- Normalize dates if possible. term should have start and end.\n"
        "- parties should include client and contractor (and subcontractor if present).\n"
        "- Return compact JSON.\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

def extract_first_json(s: str) -> Optional[dict]:
    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", s, flags=re.S|re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback: largest brace slice
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            return None
    return None

@st.cache_resource(show_spinner=True)
def load_models_cpu(base_model: str, adapter_dir: Optional[str]):
    # Threads for matmul
    torch.set_num_threads(DEFAULT_NUM_THREADS)

    # NOTE: float16 on CPU is slower but halves memory vs fp32.
    # If you hit errors on CPU with fp16, switch to torch.float32 below.
    dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        device_map={"": "cpu"} if not USE_MPS else {"": "mps"},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs).eval()

    # Optional PEFT adapter on CPU
    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, adapter_dir).eval()

    return tok, base

def free_cpu_memory(*vars_to_clear):
    for v in vars_to_clear:
        v = None
    gc.collect()

def generate_json(tok, model, prompt: str, max_new_tokens: int) -> Tuple[Optional[dict], str, float]:
    t0 = time.time()
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    device = torch.device("mps") if USE_MPS else torch.device("cpu")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.02,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    # Separate tail
    input_len = enc["input_ids"].shape[-1]
    tail = out[0][input_len:]
    text = tok.decode(tail, skip_special_tokens=True)
    parsed = extract_first_json(text)
    dt = time.time() - t0
    return parsed, text, dt

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Mistral JSON Mapper (CPU)", page_icon="🧱", layout="wide")

st.title("🧱 Mistral JSON Mapper – CPU (Invoice / SOW)")
st.caption("CPU-only inference (no bitsandbytes). Optimized for low memory usage; 7B on CPU is still heavy and slow.")

with st.sidebar:
    st.header("Model Settings")
    base_model = st.text_input("Base model", value=DEFAULT_BASE_MODEL)
    adapter_dir = st.text_input("LoRA adapter directory (optional)", value="")
    max_new = st.slider("Max new tokens", min_value=64, max_value=1024, value=DEFAULT_MAX_NEW, step=32)
    max_chars = st.slider("Max PDF chars", min_value=4000, max_value=32000, value=DEFAULT_MAX_DOC_CHARS, step=1000)
    st.markdown("**Tip:** Reduce *Max new tokens* and *Max PDF chars* if you hit memory or time limits.")

    load_btn = st.button("Load / Reload model")
    if load_btn:
        st.cache_resource.clear()

    st.divider()
    st.header("Performance Tips")
    st.write(
        "- Close other heavy apps.\n"
        "- Keep *Max new tokens* small for faster runs.\n"
        "- Upload shorter PDFs or trim with the slider.\n"
        "- If you have Apple Silicon, enabling MPS (code toggle) may help a bit."
    )

if base_model:
    with st.spinner("Loading model/tokenizer on CPU… (first time can take a while)"):
        tok, model = load_models_cpu(base_model, adapter_dir if adapter_dir.strip() else None)
else:
    st.stop()

doc_type = st.radio("Document type", options=["Invoice", "SOW"], horizontal=True)

uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)

run = st.button("Run Extraction")

if run and uploaded_files:
    results = []
    for up in uploaded_files:
        st.write("---")
        st.subheader(f"📄 {up.name}")
        try:
            # Read PDF to tmp and parse text
            tmp_path = os.path.join(st.query_params().get("_tmpdir", ["."])[0], up.name)
        except Exception:
            tmp_path = f"./{up.name}"
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())

        doc_text = read_pdf_text(tmp_path, max_chars=max_chars)
        if doc_type == "Invoice":
            prompt = build_invoice_prompt(doc_text)
        else:
            prompt = build_sow_prompt(doc_text)

        with st.spinner("Generating…"):
            js, raw, sec = generate_json(tok, model, prompt, max_new_tokens=max_new)

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Generation took **{sec:.1f}s**")
            if js is None:
                st.error("Failed to parse JSON from model output. Showing raw tail:")
                st.code(raw[-1200:])
            else:
                st.success("Parsed JSON")
                st.json(js)
                results.append({"file": up.name, "json": js})

        with col2:
            with st.expander("Show prompt (truncated)", expanded=False):
                st.code(prompt[:2000] + ("…[truncated]" if len(prompt) > 2000 else ""))

        # Light cleanup per-file
        gc.collect()

    if results:
        st.download_button(
            label="Download all results as JSONL",
            data="\n".join(json.dumps(r) for r in results),
            file_name="mapped_results.jsonl",
            mime="application/json"
        )

    # Optional: deeper cleanup when done
    if st.button("Free CPU memory (GC)"):
        free_cpu_memory(tok, model)
        st.success("Requested GC cycle executed.")

else:
    st.info("Upload PDFs and click **Run Extraction** to begin.")
