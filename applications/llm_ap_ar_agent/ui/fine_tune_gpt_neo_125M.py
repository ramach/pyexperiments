import re

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Config
MODEL_ID = "EleutherAI/gpt-neo-1.3B"
SAVE_DIR = "./gpt_neo_apar_model"

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


# Load model + tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # ✅ Set pad_token to eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).cpu()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()
st.success("✅ Base model loaded!")

# Upload training data
st.header("📁 Upload JSONL training data")
jsonl_file = st.file_uploader("Upload JSONL file with prompt/response fields", type="jsonl")

# Train button
if jsonl_file and st.button("🚀 Fine-tune"):
    lines = jsonl_file.getvalue().decode("utf-8").splitlines()
    samples = [eval(line) for line in lines]
    dataset = Dataset.from_list(samples)

    def preprocess(example):
        inputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=128)
        labels = tokenizer(example["response"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = dataset.map(preprocess)

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    with st.spinner("Training..."):
        trainer.train()
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        st.success("✅ Fine-tuning complete and model saved!")

# Inference section
st.header("🧠 Run Inference (LangChain powered)")
user_query = st.text_area("Enter AP/AR query", height=100)

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

if st.button("Generate Response"):
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    template = PromptTemplate.from_template("{query}")
    final_prompt = template.format(query=user_query)

    result = llm(final_prompt, stop=None, max_length=256)

    if result:
        st.success("✅ Model Response")
        st.write(result)
    else:
        st.warning("⚠ The model returned an empty response. Try adjusting training or prompt.")
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = hf_pipe(user_query, max_new_tokens=50, do_sample=False)
    st.write(output[0]['generated_text'])
