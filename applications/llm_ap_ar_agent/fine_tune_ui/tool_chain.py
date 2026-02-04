# toolchain.py
"""
AP toolchain adapter layer.
Each tool entrypoint accepts a JSON string and returns a Python dict result.
If a user-implemented module exists (preferred), we import and call it.
"""
from __future__ import annotations

import json
from datetime import datetime

from rule_based_invoice_verification import _to_float, _parse_date

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

def run_payment_processing_old(json_input: str):
    json_input = _coerce_json_str(json_input)
    if USER_IMPL and hasattr(USER_IMPL, "run_payment_processing"):
        return USER_IMPL.run_payment_processing(json_input)

    payload = json.loads(json_input)
    method = payload.get("payment_method") or "ACH"
    eta_days = 2 if str(method).upper() in ("ACH","WIRE") else 7
    return {"tool":"payment_processing","payment_method":method,
            "estimated_eta_days":eta_days,"timestamp":datetime.utcnow().isoformat()+"Z"}

# ---------- Payment Processing (strict gates) ----------
import json, re
from datetime import datetime, date, timedelta, timezone

def _norm_method(s: str | None) -> str:
    if not s: return "ACH"
    t = str(s).strip().lower()
    if t in ("ach", "eft"): return "ACH"
    if t in ("wire", "swift"): return "WIRE"
    if t in ("card", "credit", "credit_card", "visa", "mc", "amex"): return "CARD"
    if t in ("check", "cheque"): return "CHECK"
    return "ACH"

def _parse_terms_days(terms) -> int | None:
    if terms is None: return None
    if isinstance(terms, (int, float)): return int(terms)
    s = str(terms).strip().lower()
    if s in ("due on receipt", "immediate"): return 0
    m = re.search(r"(\d{1,3})", s)
    return int(m.group(1)) if m else None

def _business_day(d: date) -> date:
    wd = d.weekday()  # 0=Mon..6=Sun
    if wd == 5: return d + timedelta(days=2)
    if wd == 6: return d + timedelta(days=1)
    return d

def _compute_due_date(payload: dict, today: date) -> date | None:
    due_raw = payload.get("due_date") or payload.get("due")
    due = _parse_date(due_raw)
    if due:
        return due
    inv_raw = payload.get("invoice_date") or payload.get("date")
    inv = _parse_date(inv_raw)
    terms_days = _parse_terms_days(payload.get("terms"))
    if inv and terms_days is not None:
        return inv + timedelta(days=terms_days)
    return None

def _method_requirements_ok(method: str, payload: dict) -> tuple[bool, list[str]]:
    missing = []
    p = payload
    if method == "ACH":
        if not (p.get("routing_number") or p.get("bank_routing")): missing.append("routing_number")
        if not (p.get("account_number") or p.get("bank_account")): missing.append("account_number")
    elif method == "WIRE":
        if not (p.get("swift") or p.get("bic")): missing.append("swift/bic")
        if not (p.get("iban") or p.get("account_number")): missing.append("iban/account_number")
        if not (p.get("beneficiary_name") or p.get("account_name")): missing.append("beneficiary_name")
    elif method == "CARD":
        if not (p.get("card_token") or p.get("card_last4")): missing.append("card_token/last4")
        if not p.get("card_exp"): missing.append("card_exp")
    elif method == "CHECK":
        vendor = p.get("vendor") or p.get("vendor_name") or {}
        addr = None
        if isinstance(vendor, dict):
            addr = vendor.get("address") or vendor.get("mailing_address")
        else:
            addr = p.get("vendor_address") or p.get("mailing_address")
        if not addr: missing.append("vendor_mailing_address")
    return (len(missing) == 0, missing)

def run_payment_processing(json_input: str, context: dict | None = None):
    """
    Enforces:
      1) approvals_complete == True
      2) exceptions_resolved == True
      3) Valid method + required banking fields
      4) Scheduling respects due date / NET terms (next business day if overdue/none)

    context (optional):
      {
        "today": date,
        "approvals_complete": bool,
        "exceptions_resolved": bool,
        "paid_invoice_ids": set[str] | list[str],
        "block_if_missing_method": bool,   # default False (fallback to ACH)
        "force_method": "ACH"|"WIRE"|"CARD"|"CHECK"
      }
    """
    payload = json.loads(json_input) if isinstance(json_input, str) else json_input
    ctx = context or {}

    today = ctx.get("today") or datetime.now(timezone.utc).date()
    invoice_id = payload.get("invoice_id") or payload.get("invoice") or payload.get("id") or "Unknown"

    blocks = []
    issues = []

    # Duplicate payment
    known_paid = set(ctx.get("paid_invoice_ids") or [])
    if invoice_id in known_paid:
        blocks.append({"code":"duplicate_payment", "msg": f"Invoice {invoice_id} already paid."})

    # Approvals & exceptions
    approvals_complete = bool(ctx.get("approvals_complete"))
    exceptions_resolved = bool(ctx.get("exceptions_resolved", True))
    if not approvals_complete:
        blocks.append({"code":"approvals_incomplete","msg":"Approvals not completed."})
    if not exceptions_resolved:
        blocks.append({"code":"exceptions_unresolved","msg":"Exceptions remain unresolved."})

    # Amount
    amount = _to_float(payload.get("total") or payload.get("total_amount") or payload.get("amount") or payload.get("balance_due"))
    if amount is None or amount <= 0:
        blocks.append({"code":"invalid_amount","msg":"Missing or non-positive payment amount."})

    # Method normalize & required instrument fields
    method = _norm_method(ctx.get("force_method") or payload.get("payment_method"))
    if ctx.get("block_if_missing_method", False) and not (payload.get("payment_method") or ctx.get("force_method")):
        blocks.append({"code":"missing_payment_method","msg":"Payment method not provided and blocking is enabled."})
    ok, missing_fields = _method_requirements_ok(method, payload)
    if not ok:
        blocks.append({"code":"instrument_incomplete","msg":f"Missing fields for {method}: {', '.join(missing_fields)}"})

    # Due date / terms
    due = _compute_due_date(payload, today)
    if due:
        if due > today:
            scheduled = _business_day(due)
            schedule_reason = "Scheduled on due date per terms."
        else:
            scheduled = _business_day(today + timedelta(days=1))
            schedule_reason = "Invoice due today/overdue; schedule next business day."
    else:
        scheduled = _business_day(today + timedelta(days=1))
        schedule_reason = "No due date/terms; schedule next business day."

    status = "ready_to_pay" if len(blocks) == 0 else "blocked"

    return {
        "tool": "payment_processing",
        "status": status,  # "ready_to_pay" | "blocked"
        "invoice_id": invoice_id,
        "amount": amount,
        "payment_method": method,
        "scheduled_payment_date": scheduled.isoformat(),
        "schedule_reason": schedule_reason,
        "due_date": due.isoformat() if due else None,
        "blocks": blocks,   # hard stops
        "issues": issues,   # soft notes (reserved)
        "approvals_complete": approvals_complete,
        "exceptions_resolved": exceptions_resolved,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

def run_all_tools(json_input: str):
    json_input = _coerce_json_str(json_input)
    '''
    context_for_payment = {
        "today": datetime.now(timezone.utc).date(),
        "approvals_complete": (approval_out is not None and len(approval_out.get("policy_route", [])) > 0),  # or your real flag
        "exceptions_resolved": True,               # set based on your exception handling
        "paid_invoice_ids": {"INV-2024-001"},     # prevent duplicates
        # "force_method": "ACH",                   # optional override
        # "block_if_missing_method": True,         # if you want strict method presence
    }
    payment_out = run_payment_processing(json.dumps(parsed_invoice), context=context_for_payment)
    '''
    return [
        run_invoice_verification(json_input),
        run_po_matching(json_input),
        run_approval_process(json_input),
        run_payment_processing(json_input),
    ]
