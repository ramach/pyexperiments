import os
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import json
import time
import pathlib
from typing import Dict, Any, List, Tuple

import streamlit as st
import pdfplumber

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

# ---------------------------
# Helpers: PDF & chunking
# ---------------------------
def read_pdf_text(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n\n".join(text_parts).strip()

def chunk_text(text: str, max_tokens: int = 3500) -> List[str]:
    """
    Very light chunking by characters as a proxy.
    You can swap with a tokenizer-based chunker if you like.
    """
    # 1 token ~ 4 chars (rough heuristic)
    max_chars = max_tokens * 4
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

# ---------------------------
# Prompts
# ---------------------------

SYSTEM_PROMPT = """You are a careful structured data extraction assistant. 
Extract ONLY from the provided document text and follow the rules strictly.
Return a compact JSON with the exact schema and 'Missing' for fields that cannot be found by the rules.
"""

INVOICE_PROMPT = """You are given the text of a vendor invoice. Extract the following fields:

- title
- contracting_company (the buyer/bill-to)
- vendor (the seller/remit-to)
- invoice_number
- invoice_date
- due_date
- period (service period if present)
- line_items (array of {description, qty, unit_price, subtotal})
- subtotal
- tax
- total_amount
- notes

Rules:
- Do not fabricate values.
- If a field cannot be found, set it to "Missing".
- Try to normalize money values as plain numbers with decimals (e.g., 22560.00) when present.
- Return a JSON object with keys and simple values (strings/numbers), and line_items as an array of objects.
- Keep the JSON small and machine-friendly (no commentary).
Document text:
"""

SOW_PROMPT = """You are given the text of a Statement of Work (SOW). Extract:

- title
- contracting_company (the company issuing the SOW)
- vendor (the service provider)
- description_of_service
- roles_responsibilities
- line_items (if any rates/hours are listed; else empty array)
- additional_terms
- period (start/end dates if present)
- maximum_hours

Important vendor rule:
- The vendor must be present in the SOW **main line** that contains the word 'between' (e.g., "entered into by and between X and Y").
- If that 'between' main line does NOT include a vendor name, set:
  vendor = "vendor as legal contract is missing".
- Do not pull vendor from elsewhere in the doc if not present in the 'between' main line.

Other rules:
- Do not fabricate values. If a field is absent, use "Missing".
- Keep JSON small and machine-friendly (no commentary).
Document text:
"""

# ---------------------------
# Model loading (CPU + offload)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_base_and_adapter(base_model_id: str, adapter_path: str, dtype_str: str, offload_folder: str):
    """
    CPU-only load of base model + LoRA adapter with disk offload to reduce RAM.
    """
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }.get(dtype_str, torch.float32)

    os.makedirs(offload_folder, exist_ok=True)

    # Tokenizer from base
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    # Base model on CPU + disk offload
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
        offload_state_dict=True,  # ✅ NEW: avoids keeping state_dict in RAM
        trust_remote_code=True,
    )

    # Attach adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False
    )

    # eval mode
    model.eval()
    return model, tokenizer

def generate_json(
        model, tokenizer, system_prompt: str, task_prompt: str, doc_text: str,
        max_new_tokens: int = 384, temperature: float = 0.0
) -> str:
    """
    Simple generation wrapper for instruction models (Mistral Instruct style).
    Uses an Alpaca/Mistral-style chat template if available.
    """
    # Build chat if tokenizer supports chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt + "\n" + doc_text}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback simple instruction style
        prompt_text = (
            f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"### Instruction:\n{task_prompt}\n\n"
            f"### Document:\n{doc_text}\n\n"
            f"### Response (JSON only):\n"
        )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,   # keep reasonable context
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract the last JSON blob
    # (Keep minimal & robust—model should output JSON only per instructions)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        # soft validation
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # Fallback: wrap raw text
    return json.dumps({"raw": text.strip()})

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="CPU PEFT Data Mapper (PDF → JSON)", page_icon="🧩", layout="wide")
    st.title("🧩 CPU‑Only PEFT Data Mapper (PDF → JSON)")
    st.caption("Base: Mistral‑7B (regular HF format) + your LoRA adapter. CPU with disk offload—no bitsandbytes.")

    with st.sidebar:
        st.header("Model Settings")
        base_model_id = st.text_input(
            "Base model path or repo id",
            value="mistralai/Mistral-7B-Instruct-v0.2",
            help="Local folder or HF repo (must match the adapter’s base)."
        )
        adapter_path = st.text_input(
            "Adapter (LoRA) folder",
            value="./models/mistralai_trained_adapter_mapping",
            help="Folder containing adapter_config.json & adapter_model.safetensors"
        )
        dtype_str = st.selectbox("dtype", ["float32", "float16", "bfloat16"], index=0)
        offload_folder = st.text_input("Offload folder", value="./offload_cpu")
        max_new_tokens = st.slider("max_new_tokens", 128, 1024, 384, step=64)
        temperature = st.slider("temperature", 0.0, 1.0, 0.0, step=0.1)

        st.markdown("---")
        st.write("**Tips for low memory:**")
        st.write("- Keep `dtype=float32` if you see meta‑tensor errors.")
        st.write("- Close other apps; Mistral‑7B on CPU is heavy and slow.")
        st.write("- Offload folder should be on a fast SSD.")

    # Load model lazily
    if "model_bundle" not in st.session_state:
        st.session_state["model_bundle"] = None

    if st.button("Load Model + Adapter", type="primary"):
        with st.spinner("Loading base model and adapter on CPU (this can take a while)…"):
            try:
                model, tokenizer = load_base_and_adapter(base_model_id, adapter_path, dtype_str, offload_folder)
                st.session_state["model_bundle"] = (model, tokenizer)
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model/adapter: {e}")

    st.markdown("---")
    st.header("Upload PDF(s)")

    doc_type = st.radio(
        "Document type",
        options=["Invoice", "SOW"],
        horizontal=True
    )

    uploaded_files = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.session_state.get("model_bundle") is not None:
        model, tokenizer = st.session_state["model_bundle"]
        prompt = INVOICE_PROMPT if doc_type == "Invoice" else SOW_PROMPT

        if st.button(f"Extract {doc_type} JSON"):
            results: List[Tuple[str, str]] = []
            total_start = time.time()

            for f in uploaded_files:
                try:
                    text = read_pdf_text(f)
                    if not text.strip():
                        results.append((f.name, json.dumps({"error": "No text extracted"}, indent=2)))
                        continue

                    # For long docs, process first chunk (keep CPU time predictable). Expand if needed.
                    chunks = chunk_text(text, max_tokens=3500)
                    chunk = chunks[0]

                    json_str = generate_json(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=SYSTEM_PROMPT,
                        task_prompt=prompt,
                        doc_text=chunk,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )

                    # pretty print if parsable
                    try:
                        parsed = json.loads(json_str)
                        json_str = json.dumps(parsed, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                    results.append((f.name, json_str))
                except Exception as e:
                    results.append((f.name, json.dumps({"error": str(e)}, indent=2)))

            st.subheader("Results")
            for name, j in results:
                st.markdown(f"**{name}**")
                st.code(j, language="json")

            st.caption(f"Done in {time.time() - total_start:.1f}s (CPU).")

    elif uploaded_files and st.session_state.get("model_bundle") is None:
        st.info("Click **Load Model + Adapter** first.")

if __name__ == "__main__":
    main()
