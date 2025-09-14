# ---------- Payment Processing (upgraded) ----------
from __future__ import annotations

import json, re
from datetime import datetime, date, timedelta, timezone
from rule_based_invoice_verification import _to_float, _parse_date

def _norm_method(s: str | None) -> str:
    if not s: return "ACH"
    t = str(s).strip().lower()
    if t in ("ach", "eft"): return "ACH"
    if t in ("wire", "swift"): return "WIRE"
    if t in ("card", "credit", "credit_card", "visa", "mc", "amex"): return "CARD"
    if t in ("check", "cheque"): return "CHECK"
    return "ACH"

def _parse_terms_days(terms) -> int | None:
    """
    Accepts: 30, "30", "NET 30", "Net30", "n30", "due on receipt"
    Returns days or None.
    """
    if terms is None: return None
    if isinstance(terms, (int, float)): return int(terms)
    s = str(terms).strip().lower()
    if s in ("due on receipt", "immediate"): return 0
    m = re.search(r"(\d{1,3})", s)
    if m:
        try: return int(m.group(1))
        except: return None
    return None

def _business_day(d: date) -> date:
    """If weekend, move to next Monday."""
    wd = d.weekday()  # 0=Mon .. 6=Sun
    if wd == 5:  # Sat
        return d + timedelta(days=2)
    if wd == 6:  # Sun
        return d + timedelta(days=1)
    return d

def _compute_due_date(payload: dict, today: date) -> date | None:
    """
    Prefer explicit due_date; else compute from invoice_date + terms (NET n).
    """
    due_raw = payload.get("due_date") or payload.get("due")
    due = _parse_date(due_raw)
    if due:
        return due
    inv_raw = payload.get("invoice_date") or payload.get("date")
    inv = _parse_date(inv_raw)
    terms_days = _parse_terms_days(payload.get("terms"))
    if inv and terms_days is not None:
        return inv + timedelta(days=terms_days)
    # fallback: if nothing, try today (pay asap)
    return None

def _method_requirements_ok(method: str, payload: dict) -> tuple[bool, list[str]]:
    """
    Check method-specific instrument details present in payload.
    You can adapt field names to your schema.
    """
    missing = []
    p = payload

    if method == "ACH":
        # Common ACH fields
        if not (p.get("bank_routing") or p.get("routing_number")):
            missing.append("routing_number")
        if not (p.get("bank_account") or p.get("account_number")):
            missing.append("account_number")
        # Optional: account_name, bank_name
    elif method == "WIRE":
        if not (p.get("swift") or p.get("bic")):
            missing.append("swift/bic")
        if not (p.get("iban") or p.get("account_number")):
            missing.append("iban/account_number")
        if not (p.get("beneficiary_name") or p.get("account_name")):
            missing.append("beneficiary_name")
    elif method == "CARD":
        if not (p.get("card_token") or p.get("card_last4")):
            missing.append("card_token/last4")
        if not (p.get("card_exp")):
            missing.append("card_exp")
    elif method == "CHECK":
        # Need mailing address
        vendor = p.get("vendor") or p.get("vendor_name") or {}
        # vendor can be str or dict; try common fields
        addr = None
        if isinstance(vendor, dict):
            addr = vendor.get("address") or vendor.get("mailing_address")
        else:
            addr = p.get("vendor_address") or p.get("mailing_address")
        if not addr:
            missing.append("vendor_mailing_address")

    return (len(missing) == 0, missing)

def run_rule_based_payment_processing(json_input: str, context: dict | None = None):
    """
    Payment scheduling & validation.
    Preconditions (enforced here):
      - approvals_complete = True
      - exceptions_resolved = True
      - not a duplicate payment (via context['paid_invoice_ids'])

    Date logic:
      - If due_date exists and is in the future: schedule on due_date (business day)
      - If due_date is past or missing: schedule ASAP (next business day from 'today')

    Method:
      - Normalize to one of: ACH/WIRE/CARD/CHECK
      - Validate required instrument details per method

    context (optional):
      {
        "today": date,
        "approvals_complete": bool,
        "exceptions_resolved": bool,
        "paid_invoice_ids": set[str] | list[str],
        "block_if_missing_method": bool,   # default False: fallback to ACH
        "force_method": "ACH"|"WIRE"|...   # override payload method for testing/policy
      }
    """
    payload = json.loads(json_input) if isinstance(json_input, str) else json_input
    ctx = context or {}

    issues = []
    blocks = []

    today = ctx.get("today") or datetime.now(timezone.utc).date()
    invoice_id = payload.get("invoice_id") or payload.get("invoice") or payload.get("id") or "Unknown"

    # --- Duplicate payment block
    known_paid = set(ctx.get("paid_invoice_ids") or [])
    if invoice_id in known_paid:
        blocks.append({"code":"duplicate_payment", "msg": f"Invoice {invoice_id} already paid."})

    # --- Approvals & exceptions
    approvals_complete = bool(ctx.get("approvals_complete"))
    exceptions_resolved = bool(ctx.get("exceptions_resolved", True))  # default True if not provided
    if not approvals_complete:
        blocks.append({"code":"approvals_incomplete", "msg":"Approvals not completed."})
    if not exceptions_resolved:
        blocks.append({"code":"exceptions_unresolved", "msg":"Exceptions remain unresolved."})

    # --- Amount sanity
    amount = _to_float(payload.get("total") or payload.get("total_amount") or payload.get("amount") or payload.get("balance_due"))
    if amount is None or amount <= 0:
        blocks.append({"code":"invalid_amount", "msg":"Missing or non-positive payment amount."})

    # --- Method normalization & required fields
    method = _norm_method(ctx.get("force_method") or payload.get("payment_method"))
    if ctx.get("block_if_missing_method", False) and not (payload.get("payment_method") or ctx.get("force_method")):
        blocks.append({"code":"missing_payment_method","msg":"Payment method not provided and blocking is enabled."})
    ok, missing_fields = _method_requirements_ok(method, payload)
    if not ok:
        blocks.append({"code":"instrument_incomplete","msg":f"Missing fields for {method}: {', '.join(missing_fields)}"})

    # --- Due date / terms scheduling
    due = _compute_due_date(payload, today)
    if due:
        if due > today:
            scheduled = _business_day(due)
            schedule_reason = "Scheduled on due date."
        else:
            scheduled = _business_day(today + timedelta(days=1))
            schedule_reason = "Invoice overdue or due today; schedule next business day."
    else:
        scheduled = _business_day(today + timedelta(days=1))
        schedule_reason = "No due date; schedule next business day."

    # --- Final status
    can_pay = len(blocks) == 0
    status = "ready_to_pay" if can_pay else "blocked"

    return {
        "tool": "payment_processing",
        "status": status,                      # "ready_to_pay" | "blocked"
        "amount": amount,
        "payment_method": method,
        "scheduled_payment_date": scheduled.isoformat(),
        "schedule_reason": schedule_reason,
        "due_date": due.isoformat() if due else None,
        "blocks": blocks,                      # hard blockers preventing payment
        "issues": issues,                      # non-blocking notices (reserved)
        "approvals_complete": approvals_complete,
        "exceptions_resolved": exceptions_resolved,
        "invoice_id": invoice_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
