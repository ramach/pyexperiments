from docx import Document
import re

def extract_text_from_docx(docx_file) -> str:
    try:
        document = Document(docx_file) if hasattr(docx_file, "read") else Document(str(docx_file))
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        return f"Error: {e}"

def extract_invoice_data_from_docx(docx_file) -> dict:
    """
    Extract invoice-related data from a Word (.docx) document.
    :param docx_file: Path to the .docx file or a file-like object
    :return: Dictionary with extracted invoice fields
    """
    try:
        if hasattr(docx_file, "read"):
            document = Document(docx_file)
        else:
            document = Document(str(docx_file))
    except Exception as e:
        return {"error": f"Failed to open Word document: {e}"}

    text = "\n".join([para.text for para in document.paragraphs])

    # Basic regex-based parsing (customize as needed)
    invoice_data = {
        "invoice_id": _extract_field(text, r"Invoice\s*(Number|No\.?):?\s*([A-Za-z0-9\-]+)"),
        "date": _extract_field(text, r"Date[:\s]*([\d/.-]+)"),
        "vendor": _extract_field(text, r"Vendor[:\s]*([A-Za-z\s&]+)"),
        "amount": _extract_field(text, r"(Total|Amount Due|Amount)[:\s]*\$?([\d,]+\.\d{2})"),
        "raw_text": text  # for debugging or fallback
    }

    return invoice_data


def _extract_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(len(match.groups()))
    return ""
