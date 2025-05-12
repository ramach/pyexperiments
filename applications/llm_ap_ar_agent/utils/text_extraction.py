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

import pandas as pd
from typing import List, Dict, Any

import pandas as pd
from typing import Dict, Any
import re

def extract_timecard_metadata_generic(file_path: str) -> Dict[str, Any]:
    df = pd.read_excel(file_path, header=None)
    metadata = {}

    # Known field patterns (regex-style)
    field_patterns = {
        "employee_name": r"\bemployee\b",
        "manager_name": r"\bmanager\b",
        "street_address": r"street address",
        "address_2": r"address 2",
        "city_state_zip": r"city.*zip",
        "phone": r"phone",
        "email": r"e-?mail",
        "week_ending": r"week ending",
        "total_hours": r"total hours?",
        "hourly_rate": r"rate per hour",
        "total_amount": r"total pay|total amount",
    }

    def match_field(label: str) -> str:
        label_lower = label.strip().lower()
        for field, pattern in field_patterns.items():
            if re.search(pattern, label_lower):
                return field
        return None

    # Scan row-by-row
    for _, row in df.iterrows():
        row = row.fillna("").astype(str).str.strip().tolist()
        for i in range(len(row) - 1):
            field_name = match_field(row[i])
            if field_name and row[i + 1]:
                value = row[i + 1]
                # Try converting numerical fields
                if field_name in ["total_hours", "hourly_rate", "total_amount"]:
                    try:
                        value = float(re.sub(r"[^\d.]", "", value))
                    except:
                        value = 0.0
                metadata[field_name] = value

    # Build full address
    full_address = ", ".join(filter(None, [
        metadata.get("street_address", ""),
        metadata.get("address_2", ""),
        metadata.get("city_state_zip", "")
    ]))

    return {
        "employee_name": metadata.get("employee_name", "Unknown"),
        "manager_name": metadata.get("manager_name", "Unknown"),
        "phone": metadata.get("phone", "Unknown"),
        "email": metadata.get("email", "Unknown"),
        "address": full_address or "Unknown",
        "week_ending": metadata.get("week_ending", "Unknown"),
        "total_hours": metadata.get("total_hours", 0.0),
        "hourly_rate": metadata.get("hourly_rate", 0.0),
        "total_amount": metadata.get("total_amount", 0.0),
    }


