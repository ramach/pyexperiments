import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def load_finetuned_model(lora_path="./models/output_lora_tiny_mapping", base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer


def load_finetuned_model_2(lora_path="./models/output_lora_tiny"):
    # Auto-detect base model used in training
    config = PeftConfig.from_pretrained(lora_path)
    base_model_id = config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os

def load_finetuned_model_mistralai(lora_path="./models/output_lora_mistralai"):
    # Verify required files exist
    adapter_config = os.path.join(lora_path, "adapter_config.json")
    adapter_model = os.path.join(lora_path, "adapter_model.safetensors")
    if not (os.path.exists(adapter_config) and os.path.exists(adapter_model)):
        raise FileNotFoundError("LoRA adapter files are missing in the specified directory.")

    # Load LoRA config and base model ID
    config = PeftConfig.from_pretrained(lora_path)
    base_model_id = config.base_model_name_or_path
    print(f"🔍 Loading base model: {base_model_id}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,  # Efficient for large models
        device_map="auto"
    )

    # Attach LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return model, tokenizer


def run_invoice_toolchain(query: str, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response