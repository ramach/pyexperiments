import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import json

MODEL_ID = "EleutherAI/gpt-neo-125M"
SAVE_DIR = "./gpt_neo_apar_model"

# Load base model + tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding works
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()
st.success("✅ Model & tokenizer loaded!")

# Upload training data
st.header("📁 Upload Training JSONL")
jsonl_file = st.file_uploader("Upload a JSONL file with prompt/response records", type="jsonl")

if jsonl_file:
    lines = jsonl_file.getvalue().decode("utf-8").splitlines()
    data = [json.loads(line) for line in lines]
    dataset = Dataset.from_list(data)

    # Preprocess
    def preprocess(example):
        inputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(example["response"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = dataset.map(preprocess)

    if st.button("🚀 Fine-tune Model"):
        args = TrainingArguments(
            output_dir=SAVE_DIR,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="epoch"
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
        st.success("✅ Fine-tuning complete! Model saved.")

# Inference
st.header("🧠 Inference")
user_query = st.text_area("Enter your AP/AR query:")

if st.button("Generate Response"):
    try:
        hf_pipe = pipeline(
            "text-generation",
            model=SAVE_DIR,
            tokenizer=SAVE_DIR,
            pad_token_id=tokenizer.eos_token_id
        )
        output = hf_pipe(user_query, max_new_tokens=100, do_sample=False)

        if output and output[0].get("generated_text"):
            st.success("✅ Model Response")
            st.write(output[0]["generated_text"])
        else:
            st.warning("⚠ The model returned an empty response.")
    except Exception as e:
        st.error(f"❌ Error during generation: {str(e)}")
