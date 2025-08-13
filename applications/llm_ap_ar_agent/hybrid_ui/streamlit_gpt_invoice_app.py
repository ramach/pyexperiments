import streamlit as st
import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.toolchain_utils import (
    verify_invoice_fields,
    run_po_matching,
    check_approval_required,
    check_payment_eligibility,
)
from db_utils import init_db, save_invoice_and_results, fetch_all_invoices
from agents.llm_invoice_agent import map_extracted_text_to_invoice_data_with_confidence_score

init_db()
st.set_page_config(layout="wide")

def run_on_structured_data(data):
    return {
        "verification": verify_invoice_fields(data),
        "po_match": run_po_matching(data),
        "approval": check_approval_required(data),
        "payment": check_payment_eligibility(data),
    }

tabs = st.tabs(["📥 Upload & Process (GPT Mapping)", "📊 History"])

with tabs[0]:
    st.title("📄 Invoice Upload + GPT-4 Field Mapping")

    uploaded_file = st.file_uploader("Upload invoice (.txt or .pdf)", type=["txt", "pdf"])
    invoice_text = None

    if uploaded_file:
        invoice_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.text_area("Extracted Invoice Text", invoice_text, height=220)

    if invoice_text and st.button("Map with GPT-4"):
        try:
            result = map_extracted_text_to_invoice_data_with_confidence_score(invoice_text)
            st.session_state["structured"] = result
            st.session_state["file_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Mapping failed: {e}")

    if "structured" in st.session_state:
        st.subheader("✏️ Review and Correct Fields (with Confidence)")
        result = st.session_state["structured"]
        file_name = st.session_state.get("file_name", "")
        with st.form("review_form"):
            invoice_id = st.text_input("Invoice ID", value=result.get("invoice_id", {}).get("value", ""))
            invoice_date = st.text_input("Invoice Date", value=result.get("invoice_date", {}).get("value", ""))
            due_date = st.text_input("Due Date", value=result.get("due_date", {}).get("value", ""))
            vendor = st.text_input("Vendor", value=result.get("vendor", {}).get("value", ""))
            client = st.text_input("Client", value=result.get("client", {}).get("value", ""))
            total_amount = st.text_input("Total Amount", value=result.get("total_amount", {}).get("value", ""))
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

with tabs[1]:
    st.title("📊 Processed Invoice History")
    df = fetch_all_invoices()
    if df.empty:
        st.warning("No invoices processed yet.")
    else:
        st.dataframe(df, use_container_width=True)