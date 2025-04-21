# invoice_parser_utils.py
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import re
import io
from PIL import Image
from typing import Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        if text.strip():
            return text
    except Exception:
        pass
    return ""

def ocr_pdf_fallback(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path, dpi=300)
        ocr_text = []
        for img in images:
            text = pytesseract.image_to_string(img)
            ocr_text.append(text)
        return "\n".join(ocr_text)
    except Exception as e:
        return f"[OCR Fallback Error] {str(e)}"

def extract_invoice_data(text: str) -> Dict[str, str]:
    data = {}

    patterns = {
        "invoice_id": r"Invoice ID[:\s]*([A-Z0-9\-]+)",
        "vendor": r"Vendor[:\s]*(.+)",
        "date": r"Date[:\s]*(\d{4}-\d{2}-\d{2})",
        "amount": r"Amount[:\s]*\$?([\d,.]+)",
        "purchase_order": r"Purchase Order[:\s]*([A-Z0-9\-]+)",
        "payment_method": r"Payment Method[:\s]*(.+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[key] = match.group(1).strip()

    data["extracted_text"] = text.strip()
    return data
