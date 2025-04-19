# tools/invoice_tools.py
import json
import re
import sqlite3
import random

# Simulate lookup in SQLite database
DB_PATH = "data/invoices.db"

def simulate_po_lookup(po_number):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT amount FROM purchase_orders WHERE po_number = ?", (po_number,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        return f"[DB Lookup Error] {str(e)}"

def simulate_quickbooks_integration(action, data):
    # Simulate an API call to QuickBooks
    return {
        "action": action,
        "status": "success",
        "reference_id": f"QB-{random.randint(10000,99999)}",
        "data": data
    }

def wrapped_invoice_verification(data):
    try:
        #logger.debug(f"[wrapped_invoice_verification] Raw input: {data} ({type(data)})")

        if isinstance(data, dict):
            return run_invoice_verification(json.dumps(data))

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return run_invoice_verification(json.dumps(parsed))  # Ensure it's stringified again
            except json.JSONDecodeError:
                return "[Invoice Verification] Error: Expected JSON string but got invalid JSON."

        return "[Invoice Verification] Error: Unsupported input type."

    except Exception as e:
        return f"[wrapped_invoice_verification Exception] {str(e)}"

def run_invoice_verification(invoice_data):
    print(f"[DEBUG] Verifying invoice with data: {invoice_data}")
    try:
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

        po_number = invoice_data.get("purchase_order")
        amount = float(invoice_data.get("amount", 0))

        expected_amount = simulate_po_lookup(po_number)

        if expected_amount is None:
            return f"No matching PO found for {po_number}."

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

        result = simulate_quickbooks_integration("schedule_payment", {
            "vendor": vendor,
            "amount": amount,
            "method": payment_method
        })

        return f"Scheduled {payment_method} payment of ${amount} to vendor {vendor}. QuickBooks Ref: {result['reference_id']}"

    except Exception as e:
        return f"[Payment Processing Error] {str(e)}"
