import streamlit as st
import json
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_retriever import extract_business_rules_from_docx_basic
load_dotenv()

from io import BytesIO
from docx import Document
import openai

# --- Map a rule with LLM ---
def map_rule_text_to_structured(rule_text: str) -> dict:
    prompt = f"""
You are a document analyst. Analyze the rule below and return structured JSON with:
- rule_id (string or null)
- description (plain summary)
- condition (if any)
- action (what to do)
- confidence (as percentage from 0–100)

Rule:
\"\"\"
{rule_text}
\"\"\"

JSON format only.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "Failed to parse JSON", "raw": content}

# --- UI ---
st.title("📄 Business Rule Extractor (DOCX + LLM)")
st.markdown("Upload a `.docx` file with business rules. We'll extract and map them using an LLM into structured format.")

uploaded_file = st.file_uploader("Upload DOCX file", type=["docx"])

if uploaded_file:
    #get raw rules as list
    rules = extract_business_rules_from_docx_basic(uploaded_file)

    if not rules:
        st.warning("No rules detected in the document.")
    else:
        st.success(f"✅ Found {len(rules)} candidate rules")
        mapped_rules = []

        for i, rule in enumerate(rules):
            with st.expander(f"Rule {i+1} (Raw Text)"):
                st.text(rule)

                with st.spinner("Mapping rule with LLM..."):
                    mapped = map_rule_text_to_structured(rule)
                    st.json(mapped)
                    mapped_rules.append(mapped)

        if mapped_rules:
            json_output = json.dumps(mapped_rules, indent=2)
            json_bytes = BytesIO(json_output.encode("utf-8"))
            st.download_button("📥 Download Mapped Rules (JSON)", data=json_bytes, file_name="mapped_business_rules.json", mime="application/json")
