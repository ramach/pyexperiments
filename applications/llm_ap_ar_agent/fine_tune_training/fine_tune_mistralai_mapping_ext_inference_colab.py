# app.py — Mistral 7B + LoRA → JSON mapper (Invoice/SOW)
# GPU + bitsandbytes 4-bit, Toolchain integration, batch export, VRAM cleanup

import os, re, json, gc, time, io, zipfile
from typing import Optional

import torch
from pypdf import PdfReader
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llm_ap_ar_agent.fine_tune_ui.tool_chain import (
    run_invoice_verification, run_po_matching,
    run_approval_process, run_payment_processing, run_all_tools
)

# ----------------------------
# Page / defaults
# ----------------------------
st.set_page_config(page_title="Mistral JSON Mapper (GPU 4-bit) + Toolchain", page_icon="⚡", layout="wide")

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_NEW = 256
DEFAULT_MAX_DOC_CHARS = 12000

# ----------------------------
# Helpers
# ----------------------------
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
    text = text.replace("[/INST]", " ").replace("[INST]", " ")
    return text[:max_chars]

def build_invoice_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for INVOICE documents.\n"
        "Return ONLY one minified JSON object between <json> and </json>. No markdown, no explanations.\n"
        "Fields: title, invoice_id, bill_to, vendor, date, amount, supplier_information, period, terms, tax, insurance, "
        "due_date, payment_method, additional_text, line_items.\n"
        "Rules:\n"
        "- If a field is not present, set it to \"Missing\".\n"
        "- If invoice_title or title is missing, set it to \"Invoice\".\n"
        "- line_items may span lines; QTY/quantity is quantity; unit price can appear as 'unit price' or 'unitprice'.\n"
        "- 'additional notes' → additional_text. If 'Remit To' exists, treat it as vendor.\n"
        "- Split vendor/bill_to into name, address, phone, email if possible.\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

def build_sow_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for STATEMENT OF WORK (SOW) documents.\n"
        "Return ONLY one minified JSON object between <json> and </json>. No markdown, no explanations.\n"
        "Fields: title, sow_number, sow_date, agreement_date, parties, services_summary, term, roles, fees, billing, "
        "expenses, termination, contacts, signatures, additional_text.\n"
        "Rules:\n"
        "- If a field is missing, set it to \"Missing\" (or [] / {} if list/object).\n"
        "- Normalize dates if possible. term has start and end.\n"
        "- parties includes client & contractor (and subcontractor if present).\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

def extract_first_json(s: str):
    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", s, flags=re.S|re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            return None
    return None

# ----------------------------
# VRAM cleanup (meta-safe)
# ----------------------------
def free_vram_streamlit(var_names=None):
    """
    Aggressively free GPU VRAM in a Streamlit app.
    - Moves listed globals to CPU if not 'meta', then drops refs
    - Runs GC and clears CUDA caches
    """
    if var_names:
        for name in var_names:
            if name in globals() and globals()[name] is not None:
                try:
                    obj = globals()[name]
                    # Only move if not meta
                    first_param = None
                    try:
                        first_param = next(obj.parameters())
                    except Exception:
                        pass
                    if first_param is not None and getattr(first_param, "device", None) and first_param.device.type != "meta":
                        if hasattr(obj, "to"):
                            obj.to("cpu")
                except Exception:
                    pass
                globals()[name] = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

# ----------------------------
# Model loader (GPU + bnb 4-bit) — meta-safe
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_models_gpu(base_model: str, adapter_dir: Optional[str]):
    assert torch.cuda.is_available(), "CUDA/GPU not available. Switch Colab runtime to GPU."

    compute_dtype = torch.bfloat16
    try:
        _ = torch.zeros(1, dtype=compute_dtype, device="cuda")
    except Exception:
        compute_dtype = torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Avoid meta-copy issues by instantiating directly on device map
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=bnb,
        trust_remote_code=False,
        low_cpu_mem_usage=False,  # important to avoid meta + later .to()
    ).eval()

    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel
        base = PeftModel.from_pretrained(
            base,
            adapter_dir,
            device_map="auto",
            torch_dtype=compute_dtype,
            is_trainable=False,
        ).eval()

    return tok, base

def generate_json(tok, model, prompt: str, max_new_tokens: int):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
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
    input_len = enc["input_ids"].shape[-1]
    tail = out[0][input_len:]
    text = tok.decode(tail, skip_special_tokens=True)
    parsed = extract_first_json(text)
    dt = time.time() - t0
    return parsed, text, dt

# ----------------------------
# UI
# ----------------------------
st.title("⚡ Mistral JSON Mapper – GPU 4-bit (Invoice / SOW) + Toolchain")

with st.sidebar:
    st.header("Model Settings")
    base_model = st.text_input("Base model", value=DEFAULT_BASE_MODEL)
    adapter_dir = st.text_input("LoRA adapter directory (optional)", value="")
    max_new = st.slider("Max new tokens", 64, 1024, DEFAULT_MAX_NEW, 32)
    max_chars = st.slider("Max PDF chars", 4000, 32000, DEFAULT_MAX_DOC_CHARS, 1000)

    st.header("Toolchain")
    mode = st.radio("Execution mode", ["Agent (all steps)", "Targeted"], horizontal=False)
    targeted = st.multiselect(
        "Select tools (Targeted mode)",
        ["invoice_verification", "po_matching", "approval_process", "payment_processing"],
        ["invoice_verification", "po_matching", "approval_process", "payment_processing"]
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("Reload model"):
            st.cache_resource.clear()
            st.toast("Model cache cleared. Reloading on next run.", icon="♻️")
    with cols[1]:
        if st.button("Free GPU VRAM now"):
            free_vram_streamlit(["model", "tok"])
            st.toast("Requested GPU VRAM cleanup.", icon="🧹")

with st.spinner("Loading model/tokenizer on GPU (4-bit)…"):
    tok, model = load_models_gpu(base_model, adapter_dir if adapter_dir.strip() else None)
globals()["tok"] = tok
globals()["model"] = model

doc_type = st.radio("Document type", ["Invoice", "SOW"], horizontal=True)
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
run = st.button("Run Extraction")

results = []
if run and uploaded_files:
    for up in uploaded_files:
        st.write("---")
        st.subheader(f"📄 {up.name}")

        tmp = f"./{up.name}"
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())

        doc_text = read_pdf_text(tmp, max_chars=max_chars)
        prompt = (build_invoice_prompt if doc_type == "Invoice" else build_sow_prompt)(doc_text)

        with st.spinner("Generating…"):
            js, raw, sec = generate_json(tok, model, prompt, max_new_tokens=max_new)

        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Took {sec:.1f}s on GPU")
            if js is None:
                st.error("Failed to parse JSON. Raw output tail:")
                st.code(raw[-1200:])
                tool_outputs = []
            else:
                st.success("Parsed JSON")
                st.json(js)

                # ---- Run toolchain
                js_str = json.dumps(js, ensure_ascii=False)
                if mode == "Agent (all steps)":
                    tool_outputs = run_all_tools(js_str)
                else:
                    tool_outputs = []
                    if "invoice_verification" in targeted:
                        tool_outputs.append(run_invoice_verification(js_str))
                    if "po_matching" in targeted:
                        tool_outputs.append(run_po_matching(js_str))
                    if "approval_process" in targeted:
                        tool_outputs.append(run_approval_process(js_str))
                    if "payment_processing" in targeted:
                        tool_outputs.append(run_payment_processing(js_str))

                if tool_outputs:
                    tab_labels = [o.get("tool", "tool") for o in tool_outputs]
                    tabs = st.tabs([f"✅ {t}" for t in tab_labels])
                    for t, out, tab in zip(tab_labels, tool_outputs, tabs):
                        with tab:
                            st.json(out)

                results.append({"file": up.name, "parsed_json": js, "tools": tool_outputs})

        with c2:
            with st.expander("Show prompt (truncated)"):
                st.code(prompt[:2000] + ("…[truncated]" if len(prompt) > 2000 else ""))

        # light per-file cleanup
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

if results:
    # Flat JSONL
    st.download_button(
        "Download results (JSONL)",
        "\n".join(json.dumps(r) for r in results),
        "mapped_results.jsonl",
        "application/json",
    )
    # Per-file JSONs in ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            base = os.path.splitext(os.path.basename(r["file"]))[0]
            zf.writestr(f"{base}.json", json.dumps(r, indent=2))
    st.download_button(
        "Download per-file JSONs (ZIP)",
        buf.getvalue(),
        "toolchain_results.zip",
        "application/zip",
    )

if not run:
    st.info("Upload PDFs and click **Run Extraction**.")
