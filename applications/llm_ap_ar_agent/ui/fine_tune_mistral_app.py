import os
import streamlit as st

st.title("💼 Fine-Tune Mistral for AP/AR Tasks")
st.markdown("""
This app lets you:
- 📥 Download `Mistral-7B` model/tokenizer
- 🧪 Fine-tune with your JSONL dataset
- 💾 Save adapters and tokenizer locally
- 🤖 Inference from the fine-tuned model
""")

base_model_id = "mistralai/Mistral-7B-v0.1"
local_model_dir = "./mistral_base"
peft_model_dir = "./mistral-peft-apar"

# Download
if st.button("📥 Download Base Model"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(local_model_dir, exist_ok=True)

    st.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(local_model_dir)

    st.info("Downloading model (may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True,
        device_map="auto", torch_dtype="auto"
    )
    model.save_pretrained(local_model_dir)
    st.success(f"✅ Downloaded to `{local_model_dir}`")

# Fine-tuning
st.markdown("---")
st.subheader("🧪 Upload JSONL Fine-Tuning File")

jsonl_file = st.file_uploader("Upload your training dataset", type=["jsonl"])
if jsonl_file and st.button("Fine-Tune"):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
    import torch

    with open("train.jsonl", "wb") as f:
        f.write(jsonl_file.read())

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16, device_map="auto")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files="train.jsonl")["train"]
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    def preprocess(example):
        prompt = example["prompt"]
        response = example["response"]
        inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(response, truncation=True, padding="max_length", max_length=512)
        example["input_ids"] = inputs["input_ids"]
        example["labels"] = labels["input_ids"]
        return example

    dataset = dataset.map(preprocess)

    training_args = TrainingArguments(
        output_dir=peft_model_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        save_total_limit=1,
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Save adapter
    model.save_pretrained(peft_model_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    tokenizer.save_pretrained(peft_model_dir)

# Inference
st.markdown("---")
st.subheader("🤖 Inference")

user_prompt = st.text_area("Enter a prompt for the fine-tuned model")

if user_prompt and st.button("Run Inference"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)
    model = AutoModelForCausalLM.from_pretrained(peft_model_dir, torch_dtype=torch.float16, device_map="auto")

    inputs = tokenizer(user_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.text_area("🔍 Model Response", decoded)
