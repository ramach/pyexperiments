import json

def parse_llm_output_to_dict(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"raw_output": text}

def verify_invoice_fields(data):
    missing = [k for k in ["invoice_id", "invoice_date", "due_date", "vendor", "client", "total_amount"] if k not in data]
    return {"status": "OK" if not missing else "Missing fields", "missing_fields": missing}

def run_po_matching(data):
    return {"po_matched": "po_number" in data and data.get("po_number", "").startswith("PO-")}

def check_approval_required(data):
    amount = float(str(data.get("total_amount", "0")).replace("$", "").replace(",", "") or "0")
    return {"approval_required": amount > 10000, "reason": "Amount > $10,000" if amount > 10000 else "Auto-approved"}

def check_payment_eligibility(data):
    approved = data.get("approval_status", "Pending") == "Granted"
    return {"eligible_for_payment": approved, "reason": "Approved" if approved else "Pending approval"}