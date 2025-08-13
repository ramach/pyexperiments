
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

st.set_page_config(page_title="SOW Vendor Extractor", layout="wide")
st.title("📄 Statement of Work - Contracting Company & Vendor Extractor (TinyLLaMA + LoRA)")

instruction = st.text_area("Instruction", value="Extract contracting company and vendor from the statement of work sentence.")
input_text = st.text_area("Input SOW Sentence", value="Statement of Work (“SOW”) dated 26 February 2025 is entered into by and between Walsh, Brown and Miller and Anderson, Huffman and Padilla.")

run_button = st.button("🔍 Extract")

@st.cache_resource
def load_model_and_tokenizer(base_model_id: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model

if run_button:
    with st.spinner("Running inference..."):
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Adjust as needed
        adapter_path = "./models/tinyllama_trained_adapter_mapping_2"      # Adjust as needed

        tokenizer, model = load_model_and_tokenizer(base_model_id, adapter_path)

        # Construct prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("🔎 Extracted Response")
        st.code(decoded.split("### Response:")[-1].strip())
