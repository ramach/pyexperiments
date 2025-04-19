# tools/invoice_tools.py
import json
import re

def run_invoice_verification(invoice_data):
    try:
        if isinstance(invoice_data, str):
            invoice_data = json.loads(invoice_data)

        required_fields = ["invoice_id", "vendor", "amount", "date"]
        missing_fields = [field for field in required_fields if field not in invoice_data or not invoice_data[field]]

        if missing_fields:
            return f"Invoice verification failed. Missing fields: {', '.join(missing_fields)}"

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", invoice_data["date"]):
            return "Invoice verification failed. Date must be in YYYY-MM-DD format."

        return f"Invoice {invoice_data['invoice_id']} verification passed."

    except Exception as e:
        return f"[Invoice Verification Error] {str(e)}"

def run_po_matching(invoice_data):
    try:
        if isinstance(invoice_data, str):
            invoice_data = json.loads(invoice_data)

        po_database = {
            "PO-2023-A1": 1299.99,
            "PO-1002": 450.00,
            "PO-1003": 750.00
        }

        po_number = invoice_data.get("purchase_order")
        amount = float(invoice_data.get("amount", 0))

        if po_number not in po_database:
            return f"No matching PO found for {po_number}."

        expected_amount = po_database[po_number]
        if abs(expected_amount - amount) > 1e-2:
            return f"Amount mismatch for {po_number}. Expected {expected_amount}, found {amount}."

        return f"Invoice matches PO {po_number} with correct amount."

    except Exception as e:
        return f"[PO Matching Error] {str(e)}"

def run_approval_process(invoice_data):
    try:
        if isinstance(invoice_data, str):
            invoice_data = json.loads(invoice_data)

        amount = float(invoice_data.get("amount", 0))
        approvers = ["manager@company.com"]

        if amount > 1000:
            approvers.append("finance@company.com")

        return f"Invoice requires approval from: {', '.join(approvers)}"

    except Exception as e:
        return f"[Approval Process Error] {str(e)}"

def run_payment_processing(invoice_data):
    try:
        if isinstance(invoice_data, str):
            invoice_data = json.loads(invoice_data)

        payment_method = invoice_data.get("payment_method", "bank_transfer")
        vendor = invoice_data.get("vendor")
        amount = invoice_data.get("amount")

        return f"Scheduled {payment_method} payment of ${amount} to vendor {vendor}."

    except Exception as e:
        return f"[Payment Processing Error] {str(e)}"
