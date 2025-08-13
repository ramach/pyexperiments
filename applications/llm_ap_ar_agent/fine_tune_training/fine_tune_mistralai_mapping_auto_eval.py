import os, json, random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training  # <-- important

# ---------------------------
# Config
# ---------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_FILE = "invoice_synth_500.jsonl"   # JSONL with {"instruction","input","output"}
OUTPUT_DIR = "./mistral_lora_map_invoice_500"

SEED = 42
MAX_LEN = 2048
LR = 1e-4
EPOCHS = 3
BATCH = 2
GRAD_ACCUM = 8

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

def format_record(ex: Dict[str, Any]) -> str:
    """
    <s>[INST] {INSTRUCTION + DOCUMENT} [/INST] <json>{minified-json}</json></s>
    """
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = json.dumps(ex.get("output"), ensure_ascii=False, separators=(",", ":"), default=_json_default)
    user  = f"{instr}\n\nDOCUMENT:\n{inp}\n\nPlease output only minified JSON inside <json>...</json>."
    return f"<s>[INST] {user} [/INST] <json>{out}</json></s>"

def encode_and_mask(example: Dict[str, Any]) -> Dict[str, Any]:
    text = format_record(example)
    enc = tok(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,   # <-- ensure attention_mask present
    )
    input_ids = enc["input_ids"]

    # labels: mask everything BEFORE the first "<json>"
    labels = input_ids.copy()
    marker_ids = tok.encode("<json>", add_special_tokens=False)
    start = -1
    for i in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[i:i+len(marker_ids)] == marker_ids:
            start = i
            break
    if start == -1:
        labels = [-100] * len(input_ids)  # skip sample safely if no marker
    else:
        for j in range(start):
            labels[j] = -100

    enc["labels"] = labels
    return enc

@dataclass
class CausalLMJsonCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8  # good alignment on GPU

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # max len in batch (then round up to multiple)
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += (self.pad_to_multiple_of - rem)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            ids  = f["input_ids"]
            mask = f.get("attention_mask", [1] * len(ids))
            labs = f["labels"]

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids  = ids  + [pad_id] * pad_len
                mask = mask + [0] * pad_len
                labs = labs + [-100] * pad_len   # ignore loss on padding

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(mask, dtype=torch.long))
            labels.append(torch.tensor(labs, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

# ---------------------------
# Tokenizer
# ---------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---------------------------
# 4-bit base model (k-bit prep BEFORE LoRA)
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

# prepare for k-bit training (CRUCIAL)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
base.enable_input_require_grads()
base.config.use_cache = False  # required with gradient checkpointing

# ---------------------------
# LoRA wrap
# ---------------------------
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # add FFN proj if needed
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, peft_cfg)
model.train()
model.print_trainable_parameters()

# ---------------------------
# Dataset
# ---------------------------
raw = load_dataset("json", data_files={"train": TRAIN_FILE})
ds = raw["train"].map(encode_and_mask, remove_columns=raw["train"].column_names, batched=False)

# ---------------------------
# Collator
# ---------------------------
collator = CausalLMJsonCollator(tokenizer=tok, pad_to_multiple_of=8)

# ---------------------------
# Training args
# ---------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=20,
    save_steps=500,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_available(),  # True on A100/L4; otherwise False
    fp16=False,                      # keep False unless you want fp16 specifically
    optim="paged_adamw_8bit",
    report_to="none",
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=ds,
    tokenizer=tok,
)

# ---------------------------
# (Optional) quick grad diagnostic before training
# ---------------------------
if len(ds) > 0:
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, collate_fn=collator)
    batch = next(iter(dl))
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model.train()
    out = model(**batch)
    print("Sanity loss:", out.loss, "requires_grad:", out.loss.requires_grad)
    out.loss.backward()
    any_grad = any((p.grad is not None) for n, p in model.named_parameters() if p.requires_grad)
    print("Any trainable param has grad:", any_grad)

# ---------------------------
# Train & save
# ---------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Saved LoRA adapter to:", OUTPUT_DIR)
