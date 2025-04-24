import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pdf_retriever_with_image import retrieve_pdf_with_image_ocr
from agents.llm_invoice_agent import run_llm_invoice_agent, map_extracted_text_to_invoice_data_with_confidence_score

st.title("📄 Invoice & Purchase Order LLM Agent")

# --- FILE UPLOAD ---
uploaded_invoice = st.file_uploader("Upload Invoice (PDF/Image)", type=["pdf", "png", "jpg"], key="invoice_upload")
uploaded_po = st.file_uploader("Upload Purchase Order (PDF/Image)", type=["pdf", "png", "jpg"], key="po_upload")

# --- QUERY INPUT ---
query = st.selectbox("Select a Query", [
    "Analyze this invoice",
    "What is the approval process?",
    "Is this invoice valid?",
    "Match invoice with purchase order"
])

if st.button("Run Agent"):
    if not query:
        st.error("Please select a query.")
    else:
        invoice_text = ""
        po_text = ""

        # Process invoice
        if uploaded_invoice:
            invoice_text, invoice_meta, invoice_ocr = retrieve_pdf_with_image_ocr(uploaded_invoice)
            st.success("Extracted invoice text and image OCR.")
            st.text_area("🖼️ Extracted Text from Image", invoice_text, height=200)

        # Process purchase order
        if uploaded_po:
            po_text, po_meta, po_ocr = retrieve_pdf_with_image_ocr(uploaded_po)
            st.success("Extracted purchase order text and image OCR.")

        if not invoice_text and not po_text:
            st.warning("No valid input data found.")
        else:
            # Combine both invoice and PO text
            combined_text = invoice_text + "\n\n" + po_text
            mapped_data = map_extracted_text_to_invoice_data_with_confidence_score(combined_text)
            st.subheader("Mapped Invoice + PO data")
            st.json(mapped_data.get("invoice_details", mapped_data))
            input_data = {
                "extracted_text": combined_text
            }

            # Run the LLM invoice agent
            with st.spinner("Running LLM Invoice Agent..."):
                try:
                    result = run_llm_invoice_agent(query, input_data)
                    st.subheader("🔍 Agent Response")
                    st.write(result)
                except Exception as e:
                    st.error(f"Agent Error: {e}")