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
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_image
from agents.map_extracted_text_to_invoice_data_with_confidence_score import map_extracted_text_to_invoice_data_with_confidence_score
from db_utils import save_invoice_and_results, save_corrections, log_gpt_mapping, fetch_corrections_for_field

def run_on_structured_data(data: dict) -> dict:
    """
    Runs the full invoice toolchain on the provided structured invoice data.
    """
    return {
        "verification": verify_invoice_fields(data),
        "po_match": run_po_matching(data),
        "approval": check_approval_required(data),
        "payment": check_payment_eligibility(data),
    }

def build_guidance_from_corrections(corrected: dict, only_non_empty=True):
    lines = ["Use the following corrected values. Override anything conflicting."]
    for k, v in corrected.items():
        if only_non_empty and (v is None or str(v).strip() == ""):
            continue
        lines.append(f"{k} = {v}")
    return "\n".join(lines)

def is_missing(field_val: str) -> bool:
    return isinstance(field_val, str) and "MISSING" in field_val.upper()

st.set_page_config(page_title="Invoice Processor (GPT)", layout="centered")
st.title("📄 Upload Invoice → GPT Mapping → Manual Correction")

tabs = st.tabs(["📤 Upload + Mapping", "📊 View History"])

# SUGGESTIONS
with st.sidebar:
    '''
    st.markdown("🧠 **Past Field Corrections**")
    field = st.selectbox("Select field", ["invoice_id", "invoice_date", "due_date", "vendor", "client", "total_amount"])
    rows = fetch_corrections_for_field(field)
    for _, row in rows.head(3).iterrows():
        st.write(f"🧾 {row['invoice_id']}")
        st.write(f"❌ {row['original_value']}")
        st.write(f"✅ {row['corrected_value']}")
        st.markdown("---")
        '''

with tabs[0]:
    uploaded_file = st.file_uploader("Upload invoice (.txt or .pdf)", type=["txt", "pdf"])
    user_guidance = st.text_area(
        "Optional mapping guidance to improve extraction:",
        placeholder="Example: Invoice ID is TB32. Vendor is Panacea Direct Inc. Ignore footer text 'Invoice via Viewpost'."
    )

    invoice_text = None
    if uploaded_file:
        invoice_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Invoice Text", invoice_text, height=220)
        st.session_state["invoice_text_original"] = invoice_text

    if invoice_text and st.button("Map with GPT-4"):
        try:
            result = map_extracted_text_to_invoice_data_with_confidence_score(
                invoice_text,
                additional_prompt_text=user_guidance,
            )
            st.session_state["structured"] = result
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["user_guidance"] = user_guidance
            log_gpt_mapping(invoice_text, user_guidance, result)
        except Exception as e:
            st.error(f"Mapping failed: {e}")

    if "structured" in st.session_state:
        st.subheader("✏️ Review Fields (Correct as Needed)")
        result = st.session_state["structured"]
        file_name = st.session_state.get("file_name", "")
        def val(field): return result.get(field, {}).get("value", "")

        with st.form("review_form"):
            invoice_id    = st.text_input("Invoice ID",    value=val("invoice_id"))
            invoice_date  = st.text_input("Invoice Date",  value=val("invoice_date"))
            due_date      = st.text_input("Due Date",      value=val("due_date"))
            vendor        = st.text_input("Vendor",        value=val("vendor"))
            client        = st.text_input("Client",        value=val("client"))
            total_amount  = st.text_input("Total Amount",  value=val("total_amount"))

            for label, v in [("Invoice ID", invoice_id), ("Invoice Date", invoice_date),
                             ("Due Date", due_date), ("Vendor", vendor),
                             ("Client", client), ("Total Amount", total_amount)]:
                if is_missing(v):
                    st.warning(f"{label} is MISSING — please correct.")

            re_map = st.form_submit_button("Re-map with my corrections")
            run_chain = st.form_submit_button("Submit and Run Toolchain")

        if re_map:
            corrected = {
                "invoice_id": invoice_id,
                "invoice_date": invoice_date,
                "due_date": due_date,
                "vendor": vendor,
                "client": client,
                "total_amount": total_amount,
            }
            corr_guidance = build_guidance_from_corrections(corrected)
            merged_guidance = (st.session_state.get("user_guidance","").strip() + "\n" + corr_guidance).strip()
            try:
                result2 = map_extracted_text_to_invoice_data_with_confidence_score(
                    st.session_state.get("invoice_text_original", ""),
                    additional_prompt_text=merged_guidance,
                )
                st.session_state["structured"] = result2
                log_gpt_mapping(st.session_state.get("invoice_text_original", ""), merged_guidance, result2)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Re-map failed: {e}")

        if run_chain:
            structured_data = {
                "invoice_id": invoice_id,
                "invoice_date": invoice_date,
                "due_date": due_date,
                "vendor": vendor,
                "client": client,
                "total_amount": total_amount,
                "file_name": file_name,
            }
            original = {k: val(k) for k in structured_data if k != "file_name"}
            save_corrections(invoice_id, original, structured_data)
            results = run_on_structured_data(structured_data)
            #save_invoice_and_results(structured_data, results)
            st.success("✅ Toolchain completed and saved to DB")
            st.subheader("📄 Final Structured Data")
            st.json(structured_data)
            st.subheader("✅ Toolchain Results")
            st.json(results)