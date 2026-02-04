# toolchain.py
"""
AP toolchain adapter layer.
Each tool entrypoint accepts a JSON string and returns a Python dict result.
If a user-implemented module exists (preferred), we import and call it.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from rule_based_invoice_approval import run_rule_based_approval_process
from rule_based_invoice_verification import run_rule_based_invoice_verification
from rule_based_payment_processing import run_rule_based_payment_processing

#from rule_based_invoice_approval import run_rule_based_approval_process
#from rule_based_invoice_verification import run_rule_based_invoice_verification
#from rule_based_payment_processing import run_rule_based_payment_processing
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
    return run_rule_based_invoice_verification(payload)

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
    if USER_IMPL and hasattr(USER_IMPL, "run_ approval_process"):
        return USER_IMPL.run_approval_process(json_input)

    payload = json.loads(json_input)
    return run_rule_based_approval_process(payload)

def run_payment_processing(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_payment_processing"):
        return USER_IMPL.run_payment_processing(json_input)

    payload = json.loads(json_input)
    return run_rule_based_payment_processing(payload)

def run_all_tools(json_input: str):
    json_input = _coerce_json_str(json_input)
    return [
        run_invoice_verification(json_input),
        run_po_matching(json_input),
        run_approval_process(json_input),
        run_payment_processing(json_input),
    ]
