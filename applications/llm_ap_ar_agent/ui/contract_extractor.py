from docx import Document

def extract_contract_details(docx_file) -> dict:
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])

    # Very simple keyword-based parsing (can be replaced with LLM later)
    def extract_between(start, end):
        try:
            return text.split(start)[1].split(end)[0].strip()
        except IndexError:
            return "Not found"

    return {
        "Client Name and Address": extract_between("Client Name and Address:", "Consulting Firm"),
        "Consulting Firm Name and Address": extract_between("Consulting Firm Name and Address:", "Scope of Work"),
        "Scope of Work": extract_between("Scope of Work:", "Fees and Payment Terms"),
        "Fees and Payment Terms": extract_between("Fees and Payment Terms:", "Terms and Termination"),
        "Terms and Termination": extract_between("Terms and Termination:", "End of Contract")
    }
