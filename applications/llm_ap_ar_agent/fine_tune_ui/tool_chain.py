# toolchain.py
"""
AP toolchain adapter layer.
Each tool entrypoint accepts a JSON string and returns a Python dict result.
If a user-implemented module exists (preferred), we import and call it.
"""

import json
from datetime import datetime

# Try to import your real implementation (edit names as needed)
USER_IMPL = None
for mod_name in ["toolchain_impl", "ap_toolchain", "invoice_tools", "your_impl"]:
    try:
        USER_IMPL = __import__(mod_name)
        break
    except Exception:
        pass

def _coerce_json_str(data):
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)

def run_invoice_verification(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_invoice_verification"):
        return USER_IMPL.run_invoice_verification(json_input)

    payload = json.loads(json_input)
    issues = []
    required = ["title","invoice_id","date","amount","due_date","vendor","bill_to","line_items"]
    for f in required:
        if f not in payload or (isinstance(payload[f], str) and payload[f].strip() == ""):
            issues.append(f"Missing field: {f}")

    try:
        amt = float(str(payload.get("amount","0")).replace(",","").replace("$",""))
        if amt <= 0: issues.append("Amount is non-positive")
    except Exception:
        issues.append("Amount not numeric")

    items = payload.get("line_items") or []
    if not isinstance(items, list) or len(items) == 0:
        issues.append("No line items found")

    status = "valid" if not issues else "needs_review"
    return {"tool":"invoice_verification","status":status,"issues":issues,"timestamp":datetime.utcnow().isoformat()+"Z"}

def run_po_matching(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_po_matching"):
        return USER_IMPL.run_po_matching(json_input)

    payload = json.loads(json_input)
    po_id = payload.get("po_id") or payload.get("purchase_order") or "Missing"
    matched = po_id not in ("", "Missing")
    return {"tool":"po_matching","matched":matched,"po_id":po_id if matched else None,
            "details":"Exact match on PO ID" if matched else "No PO id in document",
            "timestamp":datetime.utcnow().isoformat()+"Z"}

def run_approval_process(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_approval_process"):
        return USER_IMPL.run_approval_process(json_input)

    payload = json.loads(json_input)
    amount = payload.get("amount") or "0"
    try:
        amt = float(str(amount).replace("$","").replace(",",""))
    except Exception:
        amt = 0.0
    approvers = ["AP Analyst"]
    if amt > 10000: approvers.append("Finance Manager")
    if amt > 50000: approvers.append("CFO")
    return {"tool":"approval_process","policy_route":approvers,
            "sla_days": 3 if amt<=10000 else 5 if amt<=50000 else 7,
            "timestamp":datetime.utcnow().isoformat()+"Z"}

def run_payment_processing(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_payment_processing"):
        return USER_IMPL.run_payment_processing(json_input)

    payload = json.loads(json_input)
    method = payload.get("payment_method") or "ACH"
    eta_days = 2 if str(method).upper() in ("ACH","WIRE") else 7
    return {"tool":"payment_processing","payment_method":method,
            "estimated_eta_days":eta_days,"timestamp":datetime.utcnow().isoformat()+"Z"}

def run_all_tools(json_input: str):
    json_input = _coerce_json_str(json_input)
    return [
        run_invoice_verification(json_input),
        run_po_matching(json_input),
        run_approval_process(json_input),
        run_payment_processing(json_input),
    ]
