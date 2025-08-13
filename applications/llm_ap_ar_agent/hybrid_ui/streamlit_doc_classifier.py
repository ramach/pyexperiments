import streamlit as st
import tempfile
from PyPDF2 import PdfReader
import pandas as pd
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.classify_document_text_gpt import classify_document_text_with_gpt
load_dotenv()

st.set_page_config(page_title="Document Classifier", layout="centered")
st.title("📄 AI Document Classifier")

uploaded_file = st.file_uploader("Upload a PDF or Excel document", type=["pdf", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    extracted_text = ""
    if file_extension == "pdf":
        try:
            reader = PdfReader(tmp_path)
            extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    elif file_extension == "xlsx":
        try:
            df = pd.read_excel(tmp_path, dtype=str)
            extracted_text = df.astype(str).fillna("").to_string(index=False)
        except Exception as e:
            st.error(f"Error reading Excel: {e}")

    if extracted_text:
        st.subheader("📑 Extracted Text (Preview)")
        st.text_area("Document Content", extracted_text, height=300)

        with st.spinner("Classifying..."):
            result = classify_document_text_with_gpt(extracted_text)
            st.success("✅ Classification Complete")
            st.write(result)
    else:
        st.warning("No text could be extracted from the document.")