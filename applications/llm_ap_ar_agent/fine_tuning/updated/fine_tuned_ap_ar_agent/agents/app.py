import streamlit as st
import json
from run_agent import get_agent

st.title("🧾 Fine-Tuned AP/AR Invoice Agent")

invoice_file = st.file_uploader("Upload Invoice JSON", type="json")
rules_file = st.file_uploader("Upload Business Rules (TXT)", type="txt")

if invoice_file and rules_file:
    invoice = json.load(invoice_file)
    rules = rules_file.read().decode()

    query = f"Validate this invoice: {json.dumps(invoice)} against rules: {rules}"
    agent = get_agent()

    with st.spinner("Running invoice agent..."):
        result = agent.run(query)

    st.success("Validation complete")
    st.write(result)
