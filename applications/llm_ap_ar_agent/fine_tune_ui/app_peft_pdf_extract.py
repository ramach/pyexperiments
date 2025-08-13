from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import json
import time
from io import BytesIO
from typing import Dict, Any, Optional

import streamlit as st
from pypdf import PdfReader

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

import json, re, torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stops):
        self.tokenizer = tokenizer
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        tail = text[-1200:]
        return any(s in tail for s in self.stops)

def extract_first_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end+1]
        try:
            return json.dumps(json.loads(blob), separators=(",", ":"))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.dumps(json.loads(m.group(0)), separators=(",", ":"))
    except Exception:
        return None

def clean_pdf_text(raw: str, limit=16000) -> str:
    import re
    # Strip accidental instruction markers that appear in docs
    raw = raw.replace("[INST]", "[INST_]").replace("[/INST]", "[/_INST]")
    # Basic whitespace cleanup
    txt = re.sub(r"[ \t]+", " ", raw)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()[:limit]

# ---------------------------
# PDF helpers
# ---------------------------
def read_pdf_text(file: BytesIO) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            parts.append("")
    extracted_text = "\n\n".join(parts).strip()
    st.text_area("📄 Extracted SOW Text", extracted_text, height=300)
    return extracted_text

# ---------------------------
# Prompt builders (Mistral-Instruct style)
# ---------------------------
def wrap_inst(s: str) -> str:
    return f"<s>[INST] {s.strip()} [/INST]"

def build_chat_prompt(tokenizer, system_msg: str, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    # Renders the correct prompt for this model; add_generation_prompt=True
    # appends the assistant prefix so generation starts in the right place.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def build_invoice_prompt(text: str) -> str:
    doc = clean_pdf_text(text)
    instr = (
        "You are a strict JSON extractor for INVOICE documents.\n"
        "Return ONLY a valid, minified JSON object (no markdown, no explanations).\n"
        "Fields: title, invoice_id, bill_to, vendor, date, amount, invoice_title, "
        "supplier_information, period, terms, tax, insurance, due_date, payment_method, "
        "additional_text, line_items.\n"
        "Rules:\n"
        "- If a field is not present, use \"Missing\".\n"
        "- If invoice_title is missing, set it to \"Invoice\".\n"
        "- Use \"Remit To\" as vendor if available; if vendor name missing, use business_id or client_id.\n"
        "- Split vendor and bill_to into name, address, phone, email if possible.\n"
        "- Use additional_text for anything not covered.\n"
        "- Output MUST start with '{' and end with '}'."
    )
    return f"{instr}\n\nDOCUMENT:\n{doc}"

def build_sow_prompt(text: str, tokenizer) -> str:
    system = (
        "You are a strict JSON extractor. Return ONLY one valid compact JSON object. "
        "No markdown, no explanations."
    )
    user = (
        "Extract SOW fields: title, contracting_company, vendor, description_of_service, "
        "period, roles_responsibilities, line_items, additional_terms, maximum_hours.\n\n"
        "Rule for vendor: it MUST come from the main line containing the word 'between'. "
        "If not present there, set vendor to 'vendor as legal contract is missing'.\n"
        "If a field is not present, use \"Missing\".\n"
        "Output MUST start with '{' and end with '}'.\n\n"
        f"DOCUMENT:\n{clean_pdf_text(text)}"
    )
    return build_chat_prompt(tokenizer, system, user)

def build_invoice_prompt_mistral(text: str) -> str:
    instr = (
        "You are a strict JSON extractor for INVOICE documents. "
        "Return ONLY one valid compact JSON object. No markdown, no explanations.\n"
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
    doc = clean_pdf_text(text)
    inner = f"""### Instruction:
{instr}

### Input:
{doc}

### Response:
"""
    return wrap_inst(inner)

def build_sow_prompt_mistral(text: str) -> str:
    instr = (
        "You are a strict JSON extractor for SOW documents. "
        "Return ONLY one valid compact JSON object. No markdown, no explanations.\n"
        "Extract: title, contracting_company, vendor, description_of_service, period, "
        "roles_responsibilities, line_items, additional_terms, maximum_hours.\n"
        "Rules:\n"
        "- vendor MUST come from the main line with the word 'between'. If missing in that line, set vendor to 'vendor as legal contract is missing'.\n"
        "- If a field is not present, use \"Missing\".\n"
        "Output MUST start with '{' and end with '}'."
    )
    doc = clean_pdf_text(text)
    inner = f"""### Instruction:
{instr}

### Input:
{doc}

### Response:
"""
    return wrap_inst(inner)


# ---------------------------
# JSON repair
# ---------------------------
def try_json_load(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if "{" in s and "}" in s:
        s = s[s.find("{"):s.rfind("}")+1]
    try:
        return json.loads(s)
    except Exception:
        return None


# ---------------------------
# Model loader
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_base_and_adapter(
        base_model_id: str,
        adapter_dir: str,
        load_mode: str = "4-bit (bnb)",
        dtype_choice: str = "auto",
        device_map: str = "auto",
):
    """
    Loads tokenizer + base model (quantized if chosen), then applies LoRA adapter.
    Returns (model, tokenizer).
    """
    # Sanity check: adapter config base must match user-selected base
    cfg = PeftConfig.from_pretrained(adapter_dir)
    if cfg.base_model_name_or_path != base_model_id:
        st.warning(
            f"Adapter was trained on '{cfg.base_model_name_or_path}', "
            f"but you're loading base '{base_model_id}'. This mismatch can cause KeyError or bad results."
        )

    # Quantization / dtype
    bnb_config = None
    torch_dtype = None

    if load_mode == "4-bit (bnb)":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype_choice == "bfloat16" else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_mode == "8-bit (bnb)":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # Full precision choices
        if dtype_choice == "float16":
            torch_dtype = torch.float16
        elif dtype_choice == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_choice == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None  # auto

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,            # "auto" spreads across GPUs/CPU as needed
        quantization_config=bnb_config,   # None if not using bnb
        trust_remote_code=True,           # for some Mistral builds
    )

    # Apply LoRA
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        is_trainable=False,
        device_map=device_map,           # keep consistent
    )

    model.eval()
    return model, tokenizer


import json
import re
import torch
import time

def generate_json(model, tokenizer, prompt: str, max_new_tokens=1024, temperature=0.0, top_p=0.9, n_retry=2):
    import json, re, torch, time

    # 0) sanitize: your PDF text literally contains [INST]/[/INST]—scrub them
    def _sanitize(s: str) -> str:
        return s.replace("[INST]", "[INST_]").replace("[/INST]", "[/_INST]")

    prompt = _sanitize(prompt)

    # 1) wrap to match your TinyLLaMA LoRA training style
    prompt_wrapped = f"<s>[INST] {prompt.strip()} [/INST]"

    # 2) tokenize
    enc = tokenizer(prompt_wrapped, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    input_len = input_ids.shape[-1]  # <-- length of the prompt in tokens

    parsed_json = None
    raw_output = ""

    for attempt in range(n_retry + 1):
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 3) decode ONLY the completion tail
        gen_tail_ids = out[0][input_len:]
        completion = tokenizer.decode(gen_tail_ids, skip_special_tokens=True)
        raw_output = completion.strip()

        # 4) direct parse
        try:
            return json.loads(raw_output)
        except Exception:
            pass

        # 5) try to extract/repair a JSON-like substring
        cand = _extract_json_candidate(raw_output)
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                pass

        # 6) nudge & retry
        prompt_wrapped += "\nRemember: Output must be valid minified JSON."
        enc = tokenizer(prompt_wrapped, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        input_len = input_ids.shape[-1]
        time.sleep(0.15)

    return {"error": "Failed to parse JSON from model output.", "_raw": raw_output}

def _extract_json_candidate(text: str) -> str | None:
    # Find first {...} and auto-balance if needed
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        # broader search (greedy) if needed
        m = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, flags=re.S)
        if not m:
            return None
    blob = m.group(0)
    # balance braces if truncated
    opens, closes = blob.count("{"), blob.count("}")
    if opens > closes:
        blob += "}" * (opens - closes)
    return blob

# ---------------------------
# Generation
# ---------------------------
def generate_json_mistral(model, tokenizer, prompt: str, max_new_tokens=512, temperature=0.0, top_p=1.0, n_retry=2):
    # Make sure we can pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    # Stop when JSON likely ended or model starts a new section
    stop_words = ["}\n", "\n\n", "\n###", "</json>", "RESPONSE_JSON_END"]
    stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_words)])

    for attempt in range(n_retry + 1):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,         # deterministic
                top_k=0,
                repetition_penalty=1.05, # tiny loop guard
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ✅ Prefer splitting on "### Response:" if present, otherwise on "[/INST]"
        if "### Response:" in decoded:
            completion = decoded.split("### Response:")[-1].strip()
        else:
            completion = decoded.split("[/INST]")[-1].strip()

        # Optional: show raw completion for debugging
        try:
            import streamlit as st
            st.expander("Raw completion", expanded=False).code(completion)
        except Exception:
            pass

        json_line = extract_first_json(completion)
        if json_line:
            try:
                return json.loads(json_line)
            except Exception:
                # If it’s valid JSON but not loadable due to stray chars, we’ll retry with a nudge
                pass

        # Nudge the model and retry
        prompt = prompt + "\nRemember: Return ONLY one valid minified JSON object."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    return {"error": "Failed to parse JSON from model output."}

# ---------------------------
# Simple auto-detect
# ---------------------------
def auto_detect(text: str) -> str:
    t = text.lower()
    if "invoice" in t or "remit to" in t or "invoice #" in t:
        return "Invoice"
    if "statement of work" in t or "sow" in t or "entered into by and between" in t:
        return "SOW"
    return "Invoice"


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PDF → JSON via Mistral + LoRA (PEFT)", layout="wide")
st.title("PDF → Structured JSON with Mistral (HF) + LoRA (PEFT)")

with st.sidebar:
    st.header("Model")
    base_model_id = st.text_input("Base model (HF id or local path)",
                                  value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_dir = st.text_input("LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors)",
                                value="./models/tinyllama_trained_adapter_mapping_2")

    load_mode = st.selectbox("Load mode", ["4-bit (bnb)", "8-bit (bnb)", "Full precision (no bnb)"])
    dtype_choice = st.selectbox("dtype (when not 4/8-bit)", ["auto", "float16", "bfloat16", "float32"])
    device_map = st.selectbox("device_map", ["auto", "cuda", "cpu"])

    st.header("Generation")
    temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 1024, step=128)

    st.caption("Tip: keep temperature 0.0 for strict JSON.")

model = None
tokenizer = None

col_load1, col_load2 = st.columns([1, 3])
with col_load1:
    if st.button("Load / Reload Model", type="primary"):
        try:
            model, tokenizer = load_base_and_adapter(
                base_model_id=base_model_id.strip(),
                adapter_dir=adapter_dir.strip(),
                load_mode=load_mode,
                dtype_choice=dtype_choice,
                device_map=device_map,
            )
            st.success("Model loaded.")
        except KeyError as e:
            st.error(
                f"KeyError while loading adapter into base: {e}\n\n"
                "- Ensure adapter was trained on **this** base model.\n"
                "- Verify target_modules (q_proj/v_proj) match this Mistral build.\n"
                "- Try `device_map='auto'`, `low_cpu_mem_usage=True` (already set).\n"
                "- Update `peft` & `transformers` (`pip install -U peft transformers`)."
            )
        except Exception as e:
            st.error(f"Failed to load: {e}")

with col_load2:
    st.write("")

st.subheader("Upload PDF(s)")
files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

doc_choice = st.selectbox("Document type", ["Auto (heuristic)", "Invoice", "SOW"])
run_btn = st.button("Extract")

# Keep model/tokenizer across interactions
if "model_pack" not in st.session_state:
    st.session_state.model_pack = None

if model is not None and tokenizer is not None:
    st.session_state.model_pack = (model, tokenizer)

def ensure_loaded():
    if st.session_state.model_pack is None:
        st.warning("Load the model first.")
        return None, None
    return st.session_state.model_pack

if run_btn:
    model, tokenizer = ensure_loaded()
    if model is None:
        st.stop()
    if not files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    for f in files:
        st.markdown(f"### `{f.name}`")
        try:
            text = read_pdf_text(f)
            detected = auto_detect(text) if doc_choice.startswith("Auto") else doc_choice
            st.caption(f"Detected: **{detected}**")

            prompt = build_invoice_prompt(text) if detected == "Invoice" else build_sow_prompt(text, tokenizer)

            with st.spinner("Generating JSON…"):
                data = generate_json(
                    model, tokenizer, prompt,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
            st.json(data)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
