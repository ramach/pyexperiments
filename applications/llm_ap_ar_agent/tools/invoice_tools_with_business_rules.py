from langchain.tools import Tool
import json

def invoice_verification(json_string: str) -> str:
    data = json.loads(json_string)
    invoice = data.get("invoice", {})
    issues = []

    if not invoice.get("invoice_id"):
        issues.append("Missing invoice ID.")
    if not invoice.get("vendor"):
        issues.append("Missing vendor.")
    if invoice.get("total", 0) <= 0:
        issues.append("Invalid total amount.")

    return "Invoice verification passed." if not issues else "Issues found: " + "; ".join(issues)

invoice_verification_tool = Tool(
    name="invoice_verification",
    func=invoice_verification,
    description="Verify invoice fields like ID, vendor, amount."
)

def po_validation(json_string: str) -> str:
    data = json.loads(json_string)
    invoice = data.get("invoice", {})
    po = data.get("purchase_order", {})

    if invoice.get("po_id") != po.get("po_id"):
        return "PO ID mismatch."

    if invoice.get("total") != po.get("total"):
        return "Invoice total doesn't match PO total."

    return "Purchase order validated successfully."

po_validation_tool = Tool(
    name="po_validation",
    func=po_validation,
    description="Validate invoice details against purchase order."
)

def approval_process(json_string: str) -> str:
    data = json.loads(json_string)
    invoice = data.get("invoice", {})
    rules = data.get("business_rules", {})

    threshold = rules.get("approval_threshold", 1000)
    amount = invoice.get("total", 0)

    if amount > threshold:
        return f"Approval required for amount {amount} (threshold is {threshold})."
    return "No approval required."

approval_process_tool = Tool(
    name="approval_process",
    func=approval_process,
    description="Check if invoice requires approval based on business rules."
)

def payment_processing(json_string: str) -> str:
    data = json.loads(json_string)
    invoice = data.get("invoice", {})
    rules = data.get("business_rules", {})

    payment_terms = rules.get("preferred_payment_terms", "Net 30")
    return f"Scheduled payment for invoice {invoice.get('invoice_id')} using terms {payment_terms}."

payment_processing_tool = Tool(
    name="payment_processing",
    func=payment_processing,
    description="Determine how and when to process invoice payment."
)
