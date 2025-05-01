import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"[Error extracting PDF text: {str(e)}]"

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"[Error extracting image text: {str(e)}]"

def extract_text_from_image_embedded_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Attempt to extract text directly
            text += page.get_text()

            # OCR fallback for embedded images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                text += f"\n[Image OCR Page {page_num + 1} - Image {img_index + 1}]\n{ocr_text}\n"

        return text.strip()
    except Exception as e:
        return f"[Error extracting image-embedded PDF text: {str(e)}]"

def extract_rules_from_docx(file) -> str:
    """
    Extracts rules from a .docx file and returns a structured JSON-like string.
    If no structured pattern is found, returns raw text.
    """

    if isinstance(file, BytesIO):
        doc = Document(file)
    else:
        doc = Document(BytesIO(file.read()))

    rules = {}
    raw_lines = []

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line:
            continue
        raw_lines.append(line)

        # Simple pattern: "Rule Name: Rule Description"
        if ':' in line:
            key, value = line.split(':', 1)
            rules[key.strip()] = value.strip()

    if rules:
        return json.dumps(rules, indent=2)
    else:
        # Fallback to plain text
        return "\n".join(raw_lines)

from docx import Document
from io import BytesIO
import json

def infer_type(value: str):
    """Try to cast string value to int, float, or bool."""
    value = value.strip()
    lowered = value.lower()
    if lowered in ['true', 'yes']:
        return True
    elif lowered in ['false', 'no']:
        return False
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def extract_rules_from_docx_with_type_inference(file) -> dict:
    """
    Extracts business rules from a DOCX file and infers types.
    Returns a dictionary of rules.
    """

    if isinstance(file, BytesIO):
        doc = Document(file)
    else:
        doc = Document(BytesIO(file.read()))

    rules = {}
    for para in doc.paragraphs:
        line = para.text.strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', 1)
        rules[key.strip()] = infer_type(value)

    return rules

def extract_business_rules_from_docx(file_path: str) -> list[str]:
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]



