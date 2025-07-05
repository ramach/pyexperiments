import streamlit as st
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from ..agents.invoice_agent import get_invoice_agent
from ..utils.loaders import load_invoice, load_rules

st.title("🧾 Fine-tuned AP/AR Agent")

invoice_file = st.file_uploader("Upload Invoice (JSON)", type=["json"])
rules_file = st.file_uploader("Upload Business Rules", type=["txt"])

if invoice_file and rules_file:
    invoice_data = load_invoice(invoice_file)
    rules_text = load_rules(rules_file)

    st.subheader("Invoice Preview")
    st.json(invoice_data)

    model = AutoModelForCausalLM.from_pretrained("model/", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("model/")
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    agent = get_invoice_agent(llm)

    query = st.text_input("Ask something (e.g., 'Is this invoice valid?')")

    if st.button("Run Agent"):
        output = agent.run(query)
        st.success(output)
