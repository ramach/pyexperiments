import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import json

st.set_page_config(layout="wide")
st.title("🧠 Mistral-7B Fine-Tuning and Inference (PEFT + Streamlit)")

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
SAVE_DIR = "./fine_tuned"

@st.cache_resource
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, base_model = load_base_model()
st.success("Base model and tokenizer loaded.")

st.header("📁 Upload Training Dataset (JSONL)")
jsonl_file = st.file_uploader("Upload a JSONL file (fields: 'text')", type="jsonl")

if jsonl_file:
    lines = jsonl_file.getvalue().decode("utf-8").splitlines()
    samples = [{"text": json.loads(line)["text"]} for line in lines]
    dataset = Dataset.from_list(samples)
    st.write("Sample loaded dataset:")
    st.json(samples[0])

    # PEFT config
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, config)

    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir=SAVE_DIR,
        save_total_limit=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator
    )

    if st.button("🚀 Fine-tune Model"):
        with st.spinner("Training... please wait"):
            trainer.train()
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
        st.success("✅ Fine-tuning completed and saved!")

st.header("🧪 Inference with Fine-tuned Model")
prompt = st.text_area("Enter a prompt", height=200)
if st.button("🔍 Generate"):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, SAVE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        output = pipe(prompt, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=0.7)[0]['generated_text']
        st.success("Model Output:")
        st.code(output)

    except Exception as e:
        st.error(f"Error during inference: {e}")
