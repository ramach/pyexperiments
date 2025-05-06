import streamlit as st

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_contract_extractor import extract_text_from_docx, extract_contract_fields_with_llm_experimental, extract_contract_fields_with_llm

st.title("📄 Contract Field Extractor (LLM-Powered)")

#api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📁 Upload a contract (.docx)", type="docx")

if uploaded_file:
    with st.spinner("Analyzing contract using LLM..."):
        extracted = extract_contract_fields_with_llm_experimental(uploaded_file)
    st.subheader("📌 Extracted Fields")
    st.code(extracted, language="json")

    with st.spinner("Analyzing contract using LLM..."):
        contract_text = extract_text_from_docx(uploaded_file)
        extracted = extract_contract_fields_with_llm(contract_text)
    st.subheader("📌 Extracted Fields")
    st.code(extracted, language="json")
