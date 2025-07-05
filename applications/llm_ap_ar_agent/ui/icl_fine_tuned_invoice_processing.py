import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ..utils.text_extraction import
from openai import OpenAI
import PyPDF2
import json
import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_DIR = "./flan_invoice_finetune"  # Or use pre-trained FLAN

@st.cache_resource
def load_flan_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def extract_text(pdf):
    reader = PyPDF2.PdfReader(pdf)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

hf_pipe = load_flan_pipeline()

st.title("📄 ICL Invoice Extractor")

pdf = st.file_uploader("Upload Invoice PDF", type="pdf")
model_choice = st.radio("Choose model", ["FLAN local", "OpenAI GPT-4"])

if pdf:
    text = extract_text(pdf)
    st.text_area("Extracted text", text, height=200)

    if st.button("Extract with ICL"):
        prompt = create_icl_prompt(text)
        st.code(prompt)

        if model_choice == "FLAN local":
            output = hf_pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0)[0]["generated_text"]
        else:
            client = OpenAI(api_key=OPENAI_KEY)
            output = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content

        st.code(output, language="json")

        try:
            result = json.loads(output.strip())
            st.success("✅ Extraction succeeded")
            st.json(result)
        except:
            st.error("❌ Could not parse JSON.")
            st.text(output)
