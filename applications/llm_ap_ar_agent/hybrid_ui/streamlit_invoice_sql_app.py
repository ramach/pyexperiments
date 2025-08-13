import streamlit as st
import json
import pandas as pd
from inference_utils import load_finetuned_model, run_invoice_toolchain
from toolchain_utils import (
    parse_llm_output_to_dict,
    verify_invoice_fields,
    run_po_matching,
    check_approval_required,
    check_payment_eligibility,
)
from db_utils import init_db, save_invoice_and_results, fetch_all_invoices

init_db()
st.set_page_config(layout="wide")

def run_on_structured_data(data):
    return {
        "verification": verify_invoice_fields(data),
        "po_match": run_po_matching(data),
        "approval": check_approval_required(data),
        "payment": check_payment_eligibility(data),
    }

def extract_and_prefill_form(invoice_text, model, tokenizer):
    prompt = f"Extract invoice fields as JSON:\n{invoice_text}"
    llm_response = run_invoice_toolchain(prompt, model, tokenizer)
    try:
        structured_data = parse_llm_output_to_dict(llm_response)
    except:
        structured_data = {}
    return structured_data

tabs = st.tabs(["📥 Upload & Process", "📊 History"])

# Tab 1: Hybrid Entry + Toolchain
with tabs[0]:
    st.title("🧾 Hybrid Invoice Entry: Upload + Manual")
    model, tokenizer = load_finetuned_model()

    uploaded_file = st.file_uploader("Upload invoice file (.txt or .pdf)", type=["txt", "pdf"])
    invoice_text = None

    if uploaded_file:
        invoice_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.text_area("Extracted Text", invoice_text, height=200)

    if invoice_text and st.button("Extract and Prefill"):
        extracted = extract_and_prefill_form(invoice_text, model, tokenizer)
        st.session_state["prefilled"] = extracted
        st.session_state["file_name"] = uploaded_file.name

    if "prefilled" in st.session_state:
        st.subheader("✏️ Review or Complete Fields")
        pre = st.session_state["prefilled"]
        file_name = st.session_state.get("file_name", "")
        with st.form("manual_review_form"):
            invoice_id = st.text_input("Invoice ID", value=pre.get("invoice_id", ""))
            invoice_date = st.text_input("Invoice Date", value=pre.get("invoice_date", ""))
            due_date = st.text_input("Due Date", value=pre.get("due_date", ""))
            vendor = st.text_input("Vendor", value=pre.get("vendor", ""))
            client = st.text_input("Client", value=pre.get("client", ""))
            total_amount = st.text_input("Total Amount", value=pre.get("total_amount", ""))
            submitted = st.form_submit_button("Submit and Run Toolchain")

            if submitted:
                structured_data = {
                    "invoice_id": invoice_id,
                    "invoice_date": invoice_date,
                    "due_date": due_date,
                    "vendor": vendor,
                    "client": client,
                    "total_amount": total_amount,
                    "file_name": file_name
                }
                results = run_on_structured_data(structured_data)
                save_invoice_and_results(structured_data, results)
                st.success("✅ Toolchain completed and saved to DB")
                st.subheader("📄 Final Structured Data")
                st.json(structured_data)
                st.subheader("✅ Toolchain Results")
                st.json(results)

# Tab 2: View History
with tabs[1]:
    st.title("📊 Processed Invoice History")
    df = fetch_all_invoices()
    if df.empty:
        st.warning("No invoices processed yet.")
    else:
        st.dataframe(df, use_container_width=True)