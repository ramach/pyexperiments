# ✅ Imports
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pypdf import PdfReader
import numpy as np

# === CONFIG ===
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "/content/mistral_lora_map_invoice_500"
MAX_LEN = 2048
PDF_PATH = "/content/sample_invoice.pdf"

# === Load Tokenizer ===
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Quantized Model (4-bit for GPU) ===
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    attn_implementation="eager",  # 👈 force eager mode for attention output
    quantization_config=bnb,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

# === Extract PDF Text ===
def read_pdf(path):
    r = PdfReader(path)
    text = "\n".join([p.extract_text() or "" for p in r.pages])
    return text[:16000]

doc_text = read_pdf(PDF_PATH)

# === Build Prompt ===
def wrap_inst(s): return f"<s>[INST] {s.strip()} [/INST]"

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
        "- line_items can have more than one line. Please extract all fields from line. unitprice may show as 2 words unit and price. QTY or qty or quantity is quantity.\n"
        "- \"additional notes\"  may refer to additional_text.\n"
        "- Use \"Remit To\" as vendor if available; if vendor name missing, use business_id or client_id.\n"
        "- Split vendor and bill_to into name, address, phone, email if possible.\n"
        "- Use additional_text for any content not covered by fields.\n"
        "- Output MUST start with '{' and end with '}' and be wrapped in <json>...</json>."
    )
    return wrap_inst(f"{instr}\n\nDOCUMENT:\n{doc_text}\n\nPlease output:\n<json>\n{{}}\n</json>")

# === Encode + Forward Pass ===
prompt = build_invoice_prompt(doc_text)
enc = tok(prompt, return_tensors="pt", return_attention_mask=True, truncation=True, max_length=MAX_LEN).to(model.device)
with torch.no_grad():
    outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_attentions=True)
    attn = outputs.attentions  # List of [batch, heads, q, k] per layer
    assert attn and len(attn) > 0
if not outputs.attentions:
    raise ValueError("No attentions returned from model. Check if model supports attention outputs.")

# === Compute Entropy ===
# --- SAFER ENTROPY ---
import numpy as np

def safe_entropy(attn_head_qk: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    attn_head_qk: [q_len, k_len] attention probs for one head (may contain 0/NaN due to masking)
    returns: [q_len] entropy per query token
    """
    A = attn_head_qk.astype(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = A.sum(axis=-1, keepdims=True)               # [q_len, 1]
    zero_rows = (row_sum <= eps).squeeze(-1)              # [q_len]
    A_norm = np.zeros_like(A)
    A_norm[~zero_rows] = A[~zero_rows] / np.clip(row_sum[~zero_rows], eps, None)

    A_norm = np.clip(A_norm, eps, 1.0)                    # avoid log(0)
    H = -np.sum(A_norm * np.log(A_norm), axis=-1)         # [q_len]
    H[zero_rows] = 0.0                                    # or np.nan if you prefer
    return H

def compute_entropy(attn_tensor):
    """
    attn_tensor: torch.Tensor [batch, heads, q_len, k_len] for one layer
    returns: np.ndarray [heads] mean entropy per head (avg over queries)
    """
    A = attn_tensor[0].detach().to(dtype=torch.float32, device="cpu").numpy()  # [heads, q, k]
    head_means = []
    for h in range(A.shape[0]):
        Hq = safe_entropy(A[h])          # [q_len]
        head_means.append(Hq.mean())
    return np.array(head_means)

# === Plot Entropy across Layers
# === Compute per-layer entropies (attn is a list of length L)
entropy_per_layer = [compute_entropy(a) for a in attn]

plt.figure(figsize=(12, 5))
for i, layer_entropy in enumerate(entropy_per_layer):
    plt.plot(layer_entropy, label=f"L{i}")
plt.title("Attention Entropy per Head (lower = sharper)")
plt.xlabel("Attention Head")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.show()

# === Heatmap (choose layer/head; trim Q/K for readability)
LAYER = 0
HEAD  = 0
QK_LIMIT = 128

A = attn[LAYER][0][HEAD].detach().float().cpu().numpy()   # [q, k]
A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
q_lim = min(QK_LIMIT, A.shape[0])
k_lim = min(QK_LIMIT, A.shape[1])

plt.figure(figsize=(10, 8))
sns.heatmap(A[:q_lim, :k_lim], cmap="viridis", cbar=True)
plt.title(f"Attention Heatmap (Layer {LAYER}, Head {HEAD})")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.show()
