import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import logging
import pdfplumber

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path):
    extracted_text = ""

    # Try with pdfplumber first (good for structured text and tables)
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""
    except Exception as e:
        print(f"[PDFPlumber Error] {e}")

    # If nothing was extracted or very little text, fallback to OCR
    if len(extracted_text.strip()) < 50:
        print("[OCR Fallback] Using OCR for image-based PDF")
        images = convert_from_path(file_path)
        for img in images:
            extracted_text += pytesseract.image_to_string(img)

    return extracted_text.strip()


def extract_text_from_pdf_untested(pdf_path: str) -> str:
    """Extract text from a digital PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text.strip()
    except Exception as e:
        logger.error(f"[PDF Extract] Error extracting text from digital PDF: {e}")
        return ""


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """Extract text using OCR from scanned/image-based PDF."""
    try:
        images = convert_from_path(pdf_path)
        ocr_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            ocr_text += text + "\n"
        return ocr_text.strip()
    except Exception as e:
        logger.error(f"[PDF OCR] Error extracting text from scanned PDF: {e}")
        return ""


def extract_invoice_table(pdf_path: str) -> list:
    """Attempt to extract tabular data (e.g., invoice line items) from PDF."""
    try:
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_table()
                if extracted:
                    tables.append(extracted)
        return tables
    except Exception as e:
        logger.warning(f"[PDF Table Extract] Failed: {e}")
        return []


def robust_extract_text(pdf_path: str) -> str:
    """Try digital extraction first, fall back to OCR if needed."""
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text.strip()) < 100:
        logger.info("[PDF Extract] Falling back to OCR...")
        text = extract_text_from_scanned_pdf(pdf_path)
    return text.strip()

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image file using Tesseract OCR.

    Args:
        image_path (str): Path to the image file (.png, .jpg, etc.)

    Returns:
        str: Extracted text as a string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from image: {e}")
