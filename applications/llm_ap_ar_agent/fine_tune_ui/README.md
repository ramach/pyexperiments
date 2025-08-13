# 🧾 Fine-Tuned Invoice Toolchain using LoRA & Small Language Models

This project demonstrates how to fine-tune a Small Language Model (SLM) like TinyLLaMA or Mistral using LoRA (Low-Rank Adaptation) for executing an invoice processing toolchain: **validation**, **PO match**, **approval**, and **payment readiness** — all within a lightweight, efficient pipeline.

---

## 📦 Overview

- ✅ Fine-tune a base LLM using **Supervised Fine-Tuning (SFT)** and **LoRA (PEFT)**
- 🧠 Train the model to extract structured invoice fields and reason over toolchain steps
- 🚀 Merge adapter weights for standard inference
- 🌐 Run the model in a **Streamlit UI** with **PDF upload + toolchain output**

---

## 🎯 Goals

- Structured invoice field mapping from raw text
- Decision logic for:
  - `validate_invoice`
  - `match_po`
  - `run_approval`
  - `trigger_payment`
- Low-resource fine-tuning via **LoRA**
- Deployable inference from fine-tuned SLMs

---

## 🏗️ Fine-Tuning Approach

### ✅ Method: Supervised Fine-Tuning (SFT)
- Input: Prompt-style training with instruction + input text
- Output: JSON or status classification (VALID / APPROVED / REJECTED)
- Example Format (JSONL):
```json
{
  "instruction": "Extract invoice fields from the text.",
  "input": "Invoice #3421 from ACME Corp...",
  "output": {
    "invoice_id": "3421",
    "vendor": "ACME Corp",
    "invoice_date": "2023-04-01",
    "total_amount": "$4500"
  }
}
```

---

## 🧠 What is LoRA / PEFT?

### 🔧 LoRA (Low-Rank Adaptation)
LoRA adapts only a **low-rank subspace** of weight updates:

- Instead of training all model weights, it trains two small matrices `A` and `B`
- These are injected into projection layers like `q_proj`, `v_proj`
- Enables fine-tuning with **<1%** of original model size
- Result: 🚀 Fast training, low memory, high portability

### ✅ PEFT Benefits
| Feature         | Benefit                          |
|----------------|----------------------------------|
| 🔋 Efficient     | Train with 8–16 GB VRAM          |
| 💾 Small output | Adapter size is 5–100MB          |
| ♻️ Modular       | Plug & reuse adapters on demand  |
| ⚡ Fast          | Rapid experiments / deployment  |

---
### What happens during fine-tuning
✅ fine-tuning uses gradient descent—specifically, stochastic gradient descent (SGD) or its advanced variants (like AdamW) to update the model’s parameters (or adapter weights in LoRA/PEFT). Here's how it fits:

### 🔁 What happens during fine-tuning?
For each training example:

Input (Prompt) → passed through the model to get predicted output.

Compare predicted output with the ground truth (label) using a loss function (e.g., cross-entropy loss).

Backpropagation computes gradients of the loss w.r.t. trainable parameters.

Gradient descent step updates the weights to minimize the loss:

```textmate weight = weight - learning_rate * gradient```
🎯 In LoRA fine-tuning:
Instead of updating the full base model, we update only small adapter layers (e.g., LoRA matrices A and B) using gradient descent.
This massively reduces memory and compute but still uses the same optimization steps:
```textmate Forward pass → Loss → Backward (compute gradients) → Optimizer step (gradient descent)```

🧠 Optimizer used
Most LoRA fine-tuning setups use AdamW:

```python optimizer = AdamW(model.parameters(), lr=2e-4) ```
AdamW is a variant of stochastic gradient descent with weight decay, great for transformers.

#### ✅ Summary
| Aspect	              | Used in Fine-Tuning |
|----------------------| ------------        |
| Loss function	       |  ✅ (cross-entropy or similar) |
| Backpropagation      |	✅ |
| Gradient Descent     |	✅ |
| Adam/AdamW optimizer | ✅ |
| Full model update	   | ❌ (in PEFT, only adapters are updated) |

![SGD/BackPropagation](images/fine_tune_flow.png)

![lora adapter](images/Peft.png)
### Training Method Summary
| Component     | Value                                                                  |
|---------------|------------------------------------------------------------------------|
| Training Type | **Supervised fine-tuning (SFT)**                                       |
| Model Type    | **Causal Language Model (AutoModelForCausalLM)**                       |
| Objective     | **Next-token prediction** for instruction-based prompts                |
| Label Format  | Structured JSON / classification response / tool trigger string        |
| Loss Masking  | (Optional) Use `label_ids = -100` on prompt tokens to ignore loss      |
| Tuning Method | **LoRA (PEFT)**: lightweight fine-tuning (only a few adapters trained) |
| Batch Trainer | Hugging Face `Trainer`                                                 |

## 🧪 Training Script (LoRA + Transformers)

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = AutoModelForCausalLM.from_pretrained(base_model)
model = get_peft_model(model, peft_config)
```

---

## 🔁 Merge Adapters for Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "ft-invoice-model")
model = model.merge_and_unload()
model.save_pretrained("merged-invoice-model")
```

---

## 💡 Streamlit App (Toolchain Inference)

- Upload PDF or paste invoice text
- Run toolchain prompt through fine-tuned model
- View results: validation, approval, PO match, etc.

Run it:
```bash
streamlit run invoice_toolchain_slm_pdf_app.py
```

---

## 📄 Files

| File | Purpose |
|------|---------|
| `train.jsonl` | Labeled fine-tuning dataset |
| `merge_lora_to_base_model.py` | Merge LoRA adapter to base model |
| `invoice_toolchain_slm_app.py` | Streamlit UI for manual entry |
| `invoice_toolchain_slm_pdf_app.py` | Streamlit UI with PDF support |
| `ft-invoice-model/` | Fine-tuned LoRA adapter weights |
| `merged-invoice-model/` | Fully merged model for inference |

---

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- Hugging Face PEFT: https://github.com/huggingface/peft
- TinyLLaMA: https://huggingface.co/TinyLlama

---

## ✅ Coming Next

- RAG-enhanced toolchain with vendor context
- Feedback loops with SQL audit + correction history
- Real-time PO validation using embeddings

---