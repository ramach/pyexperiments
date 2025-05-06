import fitz  # PyMuPDF
import docx
from typing import Union
from langchain.tools import tool

# --- Extractors ---

def extract_text_from_pdf(file) -> str:
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc]).strip()

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# --- Tool Chain Functions ---

@tool
def invoice_verification(data: dict) -> str:
    invoice = data.get("invoice_text", "")
    return "✅ Invoice verified." if "invoice number" in invoice.lower() else "❌ Invoice missing details."

@tool
def po_validation(data: dict) -> str:
    po = data.get("purchase_order_text", "")
    return "✅ PO validated." if "purchase order" in po.lower() else "❌ PO not valid."

@tool
def approval_process(data: dict) -> str:
    contract = data.get("contract_text", "")
    return "✅ Approval compliant." if "approval" in contract.lower() else "❌ Missing approval steps."

@tool
def payment_processing(data: dict) -> str:
    if all([data.get("invoice_text"), data.get("purchase_order_text")]):
        return "💰 Payment processing initiated."
    return "⚠️ Missing data for payment."
