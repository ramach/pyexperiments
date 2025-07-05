import re

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from langchain.llms import HuggingFacePipeline
import json

MODEL_ID = "google/flan-t5-base"  # or "google/flan-t5-small"
SAVE_DIR = "./flan_apar_model"

st.set_page_config(page_title="FLAN AP/AR LangChain Agent")

st.title("🤖 FLAN AP/AR Fine-Tuning + LangChain Integration")

def extract_amount(text):
    # Priority: $amount pattern
    match = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Fallback: number after 'for' or 'amount'
    match = re.search(r"(?:for|amount)\s*(\d+(?:,\d{3})*)", text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))

    # General fallback: any large number
    match = re.search(r"\b(\d{3,})\b", text)
    if match:
        return int(match.group(1).replace(",", ""))

    return None

# Load base model + tokenizer
@st.cache_resource
def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    return tokenizer, model

tokenizer, model = load_model_tokenizer()
st.success("✅ Base model loaded!")

# Upload training data
st.header("📁 Upload JSONL for Fine-Tuning")
jsonl_file = st.file_uploader("Upload JSONL (with prompt/response)", type="jsonl")

if jsonl_file:
    data = [json.loads(line) for line in jsonl_file.getvalue().decode().splitlines()]
    with open("tmp_train.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    dataset = load_dataset("json", data_files="tmp_train.jsonl")["train"]

    def preprocess(examples):
        inputs = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        labels = tokenizer(
            examples["response"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = dataset.map(preprocess)

    if st.button("🚀 Fine-Tune Model"):
        args = TrainingArguments(
            output_dir=SAVE_DIR,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_steps=10
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset
        )
        with st.spinner("Training in progress..."):
            trainer.train()
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
        st.success("✅ Fine-tuning complete. Model saved.")

# Inference
st.header("🧠 LangChain-Powered Inference")

user_query = st.text_area("Enter your AP/AR query:")
st.header("⚙ Business Rule Validation")
rule_max = st.number_input("Set max allowed invoice amount", min_value=100, value=10000)
if st.button("Run Business Rule Check"):
    try:
        amount = extract_amount(user_query)
        if amount:
            if amount <= rule_max:
                st.success(f"✅ Invoice amount ${amount} is within allowed max ${rule_max}")
            else:
                st.error(f"❌ Invoice amount ${amount} exceeds allowed max ${rule_max}")
        else:
            st.warning("⚠ Could not extract a valid amount from query.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if st.button("Generate LangChain Response"):
    try:
        # Load fine-tuned model + tokenizer for inference
        pipe = pipeline(
            "text2text-generation",
            model=SAVE_DIR,
            tokenizer=SAVE_DIR,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        result = llm(user_query)

        if result:
            st.success("✅ Model Response")
            st.write(result)
        else:
            st.warning("⚠ The model returned an empty response.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
