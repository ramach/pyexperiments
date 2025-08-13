import streamlit as st
from PyPDF2 import PdfReader
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from agents.map_extracted_text_to_sow_data_with_confidence_score import map_extracted_text_to_sow_data_with_confidence_score
from agents.classify_document_text_gpt import classify_document_text_with_gpt

from utils.corrections_db import init_corrections_table, insert_sow_correction, get_last_sow_correction
st.set_page_config(page_title="SOW Mapping", layout="centered")
st.title("📄 SOW Mapping with GPT-4 + Manual Guidance")
init_corrections_table()
uploaded_file = st.file_uploader("Upload SOW PDF", type=["pdf"])
extracted_text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    st.text_area("📄 Extracted SOW Text", extracted_text, height=300)

if extracted_text:
    with st.spinner("Classifying..."):
        result = classify_document_text_with_gpt(extracted_text)
        st.success("✅ Classification Complete")
        st.write(result)

    st.subheader("✏️ Review Fields (Correct or Guide as Needed)")

    if result.lower() == "sow":
        if "sow_mapping" not in st.session_state:
            st.session_state.sow_mapping = map_extracted_text_to_sow_data_with_confidence_score(extracted_text)
            if "sow_mapping" not in st.session_state:
                st.session_state.sow_mapping = map_extracted_text_to_sow_data_with_confidence_score(extracted_text)
                mapped = st.session_state.sow_mapping
        raw_company = mapped.get("contracting_company", "")
        contracting_company = str(raw_company).strip() if raw_company else ""
        form = st.form("mapping_form")
        field_updates = {}
        for key, val in st.session_state.sow_mapping.items():
            value = val.get("value") if isinstance(val, dict) else val
            field_updates[key] = form.text_input(f"{key.replace('_', ' ').title()}", value=value)
            if not contracting_company:
                st.warning("⚠️ Contracting Company could not be identified. Corrections will not be saved.")
            else:
                st.info(f"📌 Contracting Company: `{contracting_company}`")
                # 📥 Pre-fill with previous correction (if any)
                prior_guidance = get_last_sow_correction(contracting_company) if contracting_company else ""
                st.info(f"📌 Prior Guidance: `{prior_guidance}`")
                additional_prompt_text = form.text_area("🧠 Additional Prompt Guidance (optional)", value=prior_guidance)
                submit = form.form_submit_button("🔁 Re-map with My Corrections")
                if submit:
                    user_guidance = []
                    for field, updated_val in field_updates.items():
                        original_val = st.session_state.sow_mapping.get(field, {}).get("value", "")
                        if updated_val.strip() and updated_val.strip() != original_val:
                            if "please" in updated_val.lower() or "include" in updated_val.lower():
                            # treat as guidance
                            user_guidance.append(f"- {field.replace('_', ' ').title()}: {updated_val.strip()}")
                else:
                    # treat as override
                    user_guidance.append(f"- {field.replace('_', ' ').title()}: {updated_val.strip()}")
                    combined_guidance = additional_prompt_text.strip() + "\n" + "\n".join(user_guidance)
                    st.info(f"📌 Combined Guidance: `{combined_guidance}`")
                    st.session_state.sow_mapping = map_extracted_text_to_sow_data_with_confidence_score(extracted_text, combined_guidance)
                    st.success("🔁 Re-mapped using your corrections and guidance.")
                    if contracting_company and combined_guidance.strip():
                        insert_sow_correction(contracting_company, combined_guidance.strip())
                        st.success("✅ Correction saved to DB.")
                        if st.button("✅ Submit Final Mapped SOW"):
                            st.subheader("🧾 Final Mapped SOW")
                            st.json(st.session_state.sow_mapping)
                            # Save correction