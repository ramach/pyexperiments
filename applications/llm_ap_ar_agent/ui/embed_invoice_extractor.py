import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re
import json
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf, safe_json_parse, regex_extract_fields
from utils.semantic_window_mapper import semantic_field_match
# Load embedding model (cached for performance)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Define field prompts we want to extract
FIELD_LABELS = {
    "invoice_id": ["invoice number", "invoice #", "bill number"],
    "invoice_date": ["invoice date", "billed on", "billing date"],
    "due_date": ["due date", "payment due"],
    "vendor": ["vendor", "remit to", "from"],
    "client": ["client", "bill to", "to"],
    "total_amount": ["total amount", "amount due", "total"]
}

def normalize_field(text, field):
    text = text.strip()
    if field in ["invoice_date", "due_date"]:
        match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        return match.group() if match else text
    elif field == "total_amount":
        match = re.search(r"\$?[\d,]+\.\d{2}", text)
        return match.group() if match else text
    return text

def embed(texts):
    return model.encode(texts)

def extract_fields_from_text(raw_text):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    line_embeddings = embed(lines)

    results = {}
    for field, variants in FIELD_LABELS.items():
        label_embeddings = embed(variants)
        sims = cosine_similarity(line_embeddings, label_embeddings)
        best_line_idx = sims.max(axis=1).argmax()
        best_line = lines[best_line_idx]
        results[field] = normalize_field(best_line, field)

    return results

# Streamlit UI
st.title("📄 Invoice Field Extractor (PDF & Text Support)")

file = st.file_uploader("Upload an invoice (.pdf or .txt)", type=["pdf", "txt"])
raw_text = ""

if file:
    if file.name.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file)
    elif file.name.lower().endswith(".txt"):
        raw_text = file.read().decode("utf-8")

if raw_text:
    st.text_area("📜 Extracted Text", raw_text, height=250)

    if st.button("🔍 Extract Invoice Fields"):
        extracted = semantic_field_match(raw_text)
        st.subheader("🧾 Extracted Fields")
        st.json(extracted)

        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(extracted, indent=2),
            file_name="invoice_fields.json",
            mime="application/json"
        )
