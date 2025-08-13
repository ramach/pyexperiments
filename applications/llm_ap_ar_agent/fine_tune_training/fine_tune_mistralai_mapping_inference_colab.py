# %%capture
#!pip -q install "transformers>=4.41.0" "peft>=0.11.1" "accelerate>=0.30.0" "bitsandbytes>=0.43.1" pypdf

import re, json, torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === CONFIG ===
BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR  = "./mistral_lora_map"  # your trained LoRA output dir
MAX_NEW      = 768
MAX_DOC_CHARS = 18000  # cap input text to keep within context

# === Load model ===
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

def wrap_inst(s: str) -> str:
    return f"<s>[INST] {s.strip()} [/INST]"

def read_pdf_text(path: str) -> str:
    r = PdfReader(path)
    parts = []
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ""
            # Keep page markers lightly to avoid <INST> confusion
            parts.append(f"\n--- Page {i+1} ---\n{t}")
        except Exception:
            continue
    text = "\n".join(parts)
    # sanitize stray bracket sequences that might confuse chat template
    text = text.replace("[/INST]", " ").replace("[INST]", " ")
    return text[:MAX_DOC_CHARS]

def build_invoice_prompt(doc_text: str) -> str:
    instr = (
        "You are a strict JSON extractor for INVOICE documents.\n"
        "Return ONLY one minified JSON object between <json> and </json>. No markdown, no explanations.\n"
        "Fields: title, invoice_id, bill_to, vendor, date, amount, "
        "supplier_information, period, terms, tax, insurance, due_date, payment_method, "
        "additional_text, line_items.\n"
        "Rules:\n"
        "- If a field is not present, use \"Missing\".\n"
        "- If invoice_title is missing, set it to \"Invoice\".\n"
        "- line_items can have more than one line. Please extract all fields from line. unitprice may show as 2 words unit and price.\n"
        "- Use \"Remit To\" as vendor if available; if vendor name missing, use business_id or client_id.\n"
        "- Split vendor and bill_to into name, address, phone, email if possible.\n"
        "- Use additional_text for any content not covered by fields.\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

def extract_first_json(s: str):
    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", s, flags=re.S|re.I)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    # fallback: best-effort brace slice
    start = s.find("{"); end = s.rfind("}")
    if start!=-1 and end!=-1 and end>start:
        try: return json.loads(s[start:end+1])
        except: pass
    return None

def infer_invoice_from_pdf(pdf_path: str):
    doc_text = read_pdf_text(pdf_path)
    prompt = build_invoice_prompt(doc_text)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_len = enc.input_ids.shape[-1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.02,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    tail = out[0][input_len:]
    text = tok.decode(tail, skip_special_tokens=True)
    parsed = extract_first_json(text)
    return parsed, text

# === Upload + test (Colab) ===
from google.colab import files
print("Upload 1+ invoice PDFs …")
uploaded = files.upload()
for name, _ in uploaded.items():
    js, raw = infer_invoice_from_pdf(name)
    print("\n==== File:", name, "====")
    if js is None:
        print("Failed to parse JSON. Raw tail:\n", raw[-1200:])
    else:
        print(json.dumps(js, indent=2))
