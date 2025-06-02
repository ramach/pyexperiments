from datetime import datetime
from typing import Dict, Any

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

def validate_invoice(invoice: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates an invoice against dynamically supplied rules.

    Args:
        invoice (dict): The invoice data.
        rules (dict): Dictionary with rules for validation.

    Returns:
        dict: A dictionary with rule names, pass/fail status, and reasons.
    """
    results = {}

    for rule_name, rule_def in rules.items():
        # Each rule_def is expected to be a dict with "check" (a function) and "reason" (description)
        check_fn = rule_def.get("check")
        reason_fn = rule_def.get("reason")

        try:
            passed = check_fn(invoice)
            reason = reason_fn(invoice) if not passed else "OK"
        except Exception as e:
            passed = False
            reason = f"Error in rule logic: {str(e)}"

        results[rule_name] = {
            "Pass": passed,
            "Reason": reason
        }

    return results

rules = {
    "Rule 0 - Payment Terms": {
        "check": lambda inv: inv["hourly_rate"] <= 141 and
                             inv["total_hours"] <= 1976 and
                             (inv["weekly_hours"] <= 40 or inv.get("prior_approval", False)) and
                             inv.get("timesheet_approved", False),
        "reason": lambda inv: "; ".join([
            "Hourly rate exceeds $141/hr." if inv["hourly_rate"] > 141 else "",
            "Total hours exceed 1,976." if inv["total_hours"] > 1976 else "",
            "More than 40 hours/week without prior approval." if inv["weekly_hours"] > 40 and not inv.get("prior_approval", False) else "",
            "Timesheet is not client-approved." if not inv.get("timesheet_approved", False) else ""
        ]).strip("; ")
    },
    "Rule 1 - Expenses": {
        "check": lambda inv: inv["expenses"] == 0 or inv.get("receipts_attached", False),
        "reason": lambda inv: "Receipts missing for submitted expenses." if inv["expenses"] > 0 and not inv.get("receipts_attached", False) else "OK"
    },
    "Rule 2 - Invoicing & Payment Terms": {
        "check": lambda inv: abs((datetime.strptime(inv["invoice_date"], "%Y-%m-%d") -
                                  datetime.strptime(inv["last_friday_of_month"], "%Y-%m-%d")).days) <= 5 and
                             inv.get("payment_terms_days", 30) <= 30,
        "reason": lambda inv: "; ".join([
            "Invoice date is not close to last Friday of month."
            if abs((datetime.strptime(inv["invoice_date"], "%Y-%m-%d") -
                    datetime.strptime(inv["last_friday_of_month"], "%Y-%m-%d")).days) > 5 else "",
            "Payment terms exceed 30 days." if inv.get("payment_terms_days", 30) > 30 else ""
        ]).strip("; ")
    }
}

invoice_data = {
    "hourly_rate": 145.0,
    "total_hours": 1800,
    "weekly_hours": 42,
    "prior_approval": False,
    "timesheet_approved": True,
    "expenses": 500,
    "receipts_attached": False,
    "invoice_date": "2025-05-01",
    "last_friday_of_month": "2025-04-25",
    "payment_terms_days": 30
}

validation_results = validate_invoice(invoice_data, rules)
print(validation_results)

