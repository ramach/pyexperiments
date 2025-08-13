# ==== deps (Colab) ====
# !pip -q install "transformers>=4.41.0" "peft>=0.11.1" "accelerate>=0.30.0" "bitsandbytes>=0.43.1" datasets

import os, json, random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training

# ---------------------------
# Config
# ---------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_FILE = "train_invoice_sow_sample.jsonl"   # JSONL with {"instruction","input","output"}
OUTPUT_DIR = "./mistral_lora_map"

SEED = 42
MAX_LEN = 2048
LR = 1e-4
EPOCHS = 2
BATCH = 2
GRAD_ACCUM = 8
EVAL_FRACTION = 0.1   # take 10% for eval from the single file

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Helpers
# ---------------------------
def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return str(o)
    return str(o)

def build_prompt(instr: str, inp: str) -> str:
    user  = f"{instr}\n\nDOCUMENT:\n{inp}\n\nPlease output only minified JSON inside <json>...</json>."
    return f"<s>[INST] {user} [/INST] "  # note: no gold JSON appended here

def build_output_text(out_obj: Any) -> str:
    out_json = json.dumps(out_obj, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    return f"<json>{out_json}</json></s>"

def format_preview(ex: Dict[str, Any]) -> str:
    """For logging/debug — full sample text."""
    return build_prompt(ex.get("instruction","").strip(), ex.get("input","").strip()) + build_output_text(ex.get("output"))

def encode_and_mask(example: Dict[str, Any], tok, max_len: int) -> Dict[str, Any]:
    """
    Causal LM setup:
      input_ids  = prompt + output
      labels     = [-100...-100] (len(prompt)) + output_ids
      attention  = 1 across both
    We trim from the LEFT of the prompt if needed to keep full output.
    """
    instr = (example.get("instruction") or "").strip()
    inp   = (example.get("input") or "").strip()
    prompt_text = build_prompt(instr, inp)
    output_text = build_output_text(example.get("output"))

    prompt_enc = tok(prompt_text, add_special_tokens=False)
    out_enc    = tok(output_text, add_special_tokens=False)

    p_ids = prompt_enc["input_ids"]
    o_ids = out_enc["input_ids"]

    total_len = len(p_ids) + len(o_ids)
    if total_len > max_len:
        trim = total_len - max_len
        if trim < len(p_ids):
            p_ids = p_ids[trim:]  # trim prompt left
        else:
            # if output won't fit, keep last max_len tokens (rare, but safe)
            cut = max_len - 1
            p_ids = []
            o_ids = o_ids[-cut:]

    input_ids = p_ids + o_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(p_ids) + o_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

@dataclass
class CausalLMJsonCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0: max_len += (self.pad_to_multiple_of - rem)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids  = f["input_ids"]
            mask = f.get("attention_mask", [1]*len(ids))
            labs = f["labels"]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids  = ids  + [pad_id] * pad_len
                mask = mask + [0] * pad_len
                labs = labs + [-100] * pad_len
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(mask, dtype=torch.long))
            labels.append(torch.tensor(labs, dtype=torch.long))
        return {
            "input_ids": torch.stack(input_ids, 0),
            "attention_mask": torch.stack(attention_mask, 0),
            "labels": torch.stack(labels, 0),
        }

# ---------------------------
# Tokenizer
# ---------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = MAX_LEN

# ---------------------------
# 4-bit base + k-bit prep
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
base.enable_input_require_grads()
base.config.use_cache = False

# LoRA
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,  # a bit more dropout to regularize
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, peft_cfg)
model.train()
model.print_trainable_parameters()

# ---------------------------
# Dataset: split from one file
# ---------------------------
#raw_all = load_dataset("json", data_files={"all": TRAIN_FILE})["all"]
raw_all = load_dataset("json", data_files={"train": TRAIN_FILE})
rows = list(raw_all)
random.shuffle(rows)
n_eval = max(50, int(len(rows) * EVAL_FRACTION)) if len(rows) > 200 else max(10, int(len(rows)*0.1))
eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

def map_fn(ex):
    return encode_and_mask(ex, tok, MAX_LEN)

train_ds = Dataset.from_list(train_rows).map(map_fn, batched=False)
eval_ds  = Dataset.from_list(eval_rows).map(map_fn, batched=False)

print(f"Split -> train={len(train_ds)}  eval={len(eval_ds)}")

# Collator
collator = CausalLMJsonCollator(tokenizer=tok, pad_to_multiple_of=8)

# ---------------------------
# Training args (no eval scheduling args to avoid version issues)
# ---------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,                 # mild regularization
    logging_steps=20,
    save_steps=500,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_available(),
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,  # we’ll call evaluate() manually
    tokenizer=tok,         # FutureWarning-safe for now
)

# quick grad sanity
if len(train_ds) > 0:
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=1, collate_fn=collator)
    b = next(iter(dl))
    b = {k: v.to(model.device) for k, v in b.items()}
    out = model(**b)
    print("Sanity loss:", out.loss, "requires_grad:", out.loss.requires_grad)
    out.loss.backward()
    any_grad = any((p.grad is not None) for n,p in model.named_parameters() if p.requires_grad)
    print("Any trainable param has grad:", any_grad)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Saved LoRA adapter to:", OUTPUT_DIR)

# ---------------------------
# Manual eval (post-train)
# ---------------------------
metrics = trainer.evaluate()
print("Eval metrics:", metrics)

# Tiny generation preview on 2 eval samples
def gen_preview(example, max_new_tokens=256):
    prompt = build_prompt(example.get("instruction","").strip(), example.get("input","").strip())
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
            repetition_penalty=1.02,
        )
    text = tok.decode(out[0], skip_special_tokens=True)[-1200:]
    return text

print("\n=== Sample generations ===")
for i in range(min(2, len(eval_rows))):
    print(gen_preview(eval_rows[i]))
    print("---")
