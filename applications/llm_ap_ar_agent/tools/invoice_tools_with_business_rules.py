from langchain.tools import Tool
import json

def invoice_verification(input_data: dict) -> str:
    invoice = input_data.get("invoice_and_PO_details", {})
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

def po_validation(input_data: dict) -> str:
    invoice = input_data.get("invoice_and_PO_details", {})
    if invoice.get("purchase_order") is None:
        return "Purchase order missing."
    return "Purchase order validated successfully."

po_validation_tool = Tool(
    name="po_validation",
    func=po_validation,
    description="Validate invoice details against purchase order."
)

def approval_process(input_data: dict) -> str:
    data = input_data.get("invoice_and_PO_details", {})
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

def payment_processing(input_data: dict) -> str:
    input_data.get("invoice_and_PO_details", {})
    invoice = input_data.get("invoice", {})
    rules = input_data.get("business_rules", {})

    payment_terms = rules.get("preferred_payment_terms", "Net 30")
    return f"Scheduled payment for invoice {invoice.get('invoice_id')} using terms {payment_terms}."

payment_processing_tool = Tool(
    name="payment_processing",
    func=payment_processing,
    description="Determine how and when to process invoice payment."
)
