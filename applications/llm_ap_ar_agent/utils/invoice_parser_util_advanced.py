import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path):
    """
    Attempts native text extraction from a PDF.
    Falls back to OCR using Tesseract if native extraction is insufficient.
    """
    extracted_text = ""

    try:
        # Attempt native text extraction with PyMuPDF
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            if text:
                extracted_text += text + "\n"
        doc.close()

        if extracted_text.strip():
            return extracted_text.strip()
    except Exception as e:
        print(f"[PDF Extractor] Native text extraction failed: {e}")

    # Fallback to OCR using pytesseract
    try:
        print("[PDF Extractor] Falling back to OCR...")
        images = convert_from_path(pdf_path)
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text += text + "\n"
    except Exception as e:
        print(f"[PDF Extractor] OCR extraction failed: {e}")

    return extracted_text.strip()
