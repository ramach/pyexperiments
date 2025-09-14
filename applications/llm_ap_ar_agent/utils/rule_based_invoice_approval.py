# ---------- Invoice Approval (Section 11 rules) ----------
from __future__ import annotations

import json
from datetime import datetime
from rule_based_invoice_verification import _to_float

def _has_po(payload: dict) -> bool:
    return bool(
        (payload.get("po_id") or payload.get("purchase_order") or payload.get("po"))
        and str(payload.get("po_id") or payload.get("purchase_order") or payload.get("po")).strip().lower() not in ("missing", "none", "")
    )

def _line_items(payload: dict):
    items = payload.get("line_items") or []
    return items if isinstance(items, list) else []

def _is_capex(item: dict) -> bool:
    """
    Heuristics to classify a line as Capex. You can tighten this to your schema.
    """
    txt = " ".join(str(item.get(k, "")) for k in ("category","type","capex","description","desc","gl_code")).lower()
    flags = [
        "capex", "capital", "capital expenditure", "asset",
        "equipment", "hardware", "capitalized", "fixed asset"
    ]
    return any(f in txt for f in flags) or (str(item.get("capex", "")).strip().lower() in ("true","yes","1"))

def _capex_amount_over_threshold(items, threshold=5000.0):
    """
    Returns True if any single Capex line item amount > threshold.
    Uses explicit 'amount' if present, else qty*unit_price.
    """
    for it in items:
        if not _is_capex(it):
            continue
        amt = _to_float(it.get("amount"))
        if amt is None:
            q = _to_float(it.get("qty") or it.get("quantity"))
            r = _to_float(it.get("unit_price") or it.get("unit price") or it.get("rate") or it.get("bill rate") or it.get("bill_rate"))
            if q is not None and r is not None:
                amt = q * r
        if amt is not None and amt > threshold:
            return True
    return False

def _uniq_stable(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def run_rule_based_approval_process(json_input: str, context: dict | None = None):
    """
    Section 11: Approval policy
    - Always Finance Maker -> Finance Checker first
    - If PO exists, apply PO approval rules; if no PO, Finance Director must approve
    - Apply Vendor Mgmt, SOW, Contract approval rules
    - Amount thresholds:
        > 5,000   => Finance Director
        > 25,000  => Finance Vice President
        > 50,000  => CEO
    - Capex line item > 5,000 => Treasurer
    - If exceptions before payment => Finance Director
    context (optional) can carry:
      - has_exceptions: bool
      - external_approvals: { "vendor": "approved|pending|required", "po": "...", "sow": "...", "contract": "..." }
      - today, business_unit, etc. (for your own routing extensions)
    """
    context = context or {}
    payload = json.loads(json_input) if isinstance(json_input, str) else json_input

    amount = _to_float(payload.get("total") or payload.get("total_amount") or payload.get("amount") or payload.get("balance_due"))
    po_exists = _has_po(payload)
    items = _line_items(payload)

    # --- Base chain: maker then checker
    route = [
        "Finance Maker",
        "Finance Checker",
    ]
    reasons = [
        {"approver":"Finance Maker",   "reason":"All invoices must be prepared by maker."},
        {"approver":"Finance Checker", "reason":"All invoices must be reviewed by checker."},
    ]

    # --- PO path vs No-PO path
    if po_exists:
        route.append("PO Approval (per policy)")
        reasons.append({"approver":"PO Approval (per policy)","reason":"Purchase order exists; apply PO approval rules."})
    else:
        route.append("Finance Director")
        reasons.append({"approver":"Finance Director","reason":"No purchase order present; director approval required."})

    # --- Apply Vendor / SOW / Contract approval rules (gateways)
    route.extend(["Vendor Management Approval (per policy)",
                  "SOW Approval (per policy)",
                  "Contract Approval (per policy)"])
    reasons.extend([
        {"approver":"Vendor Management Approval (per policy)","reason":"Apply vendor management approval rules."},
        {"approver":"SOW Approval (per policy)","reason":"Apply statement of work approval rules."},
        {"approver":"Contract Approval (per policy)","reason":"Apply contract approval rules."},
    ])

    # --- Threshold approvals (strictly additional)
    if amount is not None:
        if amount > 5000:
            route.append("Finance Director")
            reasons.append({"approver":"Finance Director","reason":"Invoice amount > $5,000."})
        if amount > 25000:
            route.append("Finance Vice President")
            reasons.append({"approver":"Finance Vice President","reason":"Invoice amount > $25,000."})
        if amount > 50000:
            route.append("Chief Executive Officer")
            reasons.append({"approver":"Chief Executive Officer","reason":"Invoice amount > $50,000."})

    # --- Capex line item rule
    if _capex_amount_over_threshold(items, threshold=5000.0):
        route.append("Treasurer")
        reasons.append({"approver":"Treasurer","reason":"Capex line item > $5,000."})

    # --- Exceptions before payment
    if context.get("has_exceptions"):
        route.append("Finance Director")
        reasons.append({"approver":"Finance Director","reason":"Exceptions exist prior to payment scheduling."})

    # --- Normalize (dedupe while preserving order)
    route = _uniq_stable(route)

    # --- External approvals status passthrough (optional)
    external = context.get("external_approvals") or {}
    policy = {"tool": "approval_process", "policy_route": route, "reasons": reasons, "amount": amount,
              "po_exists": po_exists, "capex_flag": _capex_amount_over_threshold(items, 5000.0),
              "exceptions_flag": bool(context.get("has_exceptions")), "external": external,
              "timestamp": datetime.utcnow().isoformat() + "Z",
              "sla_days": 3 if (amount or 0) <= 10000 else 5 if (amount or 0) <= 50000 else 7}
    # Simple SLA suggestion (you can tune this)

    return policy
'''
res = run_approval_process(
    json.dumps(parsed_invoice),
    context={
        "has_exceptions": True,
        "external_approvals": {"vendor": "pending", "po": "approved", "sow": "required", "contract": "approved"}
    }

)
'''

