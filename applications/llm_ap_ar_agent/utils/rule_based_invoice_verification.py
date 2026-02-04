# ---------- Invoice Validation (rules-based) ----------
import json, re
from datetime import datetime, timezone, timedelta

NUMERIC_RE = re.compile(r"[-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?")

def _to_float(x):
    """Robust money/number parser: '$22,560.00' -> 22560.0 ; '1,234.5' -> 1234.5 ; None -> None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    # Prefer last number in the string (handles 'USD 1,234.00 (estimated)')
    m = list(NUMERIC_RE.finditer(s.replace(",", "")))
    if not m:
        return None
    try:
        return float(m[-1].group(0))
    except Exception:
        return None

def _parse_date(s):
    """Try several common formats; returns naive UTC date() or None."""
    if not s:
        return None
    s = str(s).strip()
    fmts = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y",
        "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            pass
    # lenient: pull YYYY-MM-DD first
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except Exception:
            pass
    return None

def _get_first(payload, *keys):
    """Fetch first non-empty value across synonyms."""
    for k in keys:
        if k in payload and payload[k] not in (None, "", "Missing"):
            return payload[k]
    return None

def _collect_line_items(payload):
    """Return normalized line items: [{'qty': float|None, 'unit_price': float|None, 'amount': float|None, 'desc': str}]"""
    items = payload.get("line_items") or []
    norm = []
    for it in items if isinstance(items, list) else []:
        qty = _get_first(it, "qty", "quantity", "QTY", "Quantity")
        rate = _get_first(it, "unit_price", "unit price", "unitprice", "rate", "bill rate", "bill_rate")
        amt  = _get_first(it, "amount", "subtotal", "line_total", "line amount")
        desc = _get_first(it, "description", "desc", "item", "line")
        norm.append({
            "qty": _to_float(qty),
            "unit_price": _to_float(rate),
            "amount": _to_float(amt),
            "desc": desc or ""
        })
    return norm

def _sum_line_items(items):
    """Prefer explicit line amounts; else compute qty*unit_price."""
    total = 0.0
    any_amount = any(i.get("amount") is not None for i in items)
    if any_amount:
        for i in items:
            v = i.get("amount")
            if v is not None:
                total += float(v)
    else:
        for i in items:
            q = i.get("qty"); r = i.get("unit_price")
            if q is not None and r is not None:
                total += float(q) * float(r)
    return total

def _status_from_issues(issues):
    # duplicates or invalid format -> rejected, else needs_review if any issue, else valid
    hard = any(i.get("code") in ("duplicate_invoice_id", "invalid_format") for i in issues)
    if hard:
        return "rejected"
    return "needs_review" if issues else "valid"

def _run_invoice_verification(payload: dict, context: dict | None = None) -> dict:
    """
    Validates invoice JSON per business rules (Sections 1, 6).
    payload: parsed invoice JSON (from extraction step)
    context: optional {
        'known_invoice_ids': set[str],
        'today': date,
        'source_format': 'pdf'|'docx'  # if not already in payload
    }
    """
    context = context or {}
    issues = []
    checks = {}
    computed = {}

    # ---- Section 1: Format requirement (pdf/docx)
    source_format = _get_first(payload, "source_format") or context.get("source_format")
    if source_format:
        fmt_ok = str(source_format).lower() in ("pdf", "docx")
        checks["format_ok"] = fmt_ok
        if not fmt_ok:
            issues.append({"code":"invalid_format","msg":f"Unsupported format: {source_format}. Require PDF or DOCX."})
    else:
        checks["format_ok"] = None  # unknown; not a hard fail

    # Required fields (Section 1 — core set)
    invoice_id  = _get_first(payload, "invoice_id", "invoice", "id")
    invoice_dt  = _get_first(payload, "invoice_date", "date")
    due_dt      = _get_first(payload, "due_date", "due")
    vendor_name = _get_first(payload, "vendor", "vendor_name", "remit_to", "supplier", "supplier_information")
    total_amt   = _get_first(payload, "total", "total_amount", "amount", "balance_due")

    req_missing = []
    for nm, val in [("invoice_id", invoice_id), ("invoice_date", invoice_dt),
                    ("due_date", due_dt), ("vendor_name", vendor_name),
                    ("total_amount", total_amt)]:
        if val in (None, "", "Missing"):
            req_missing.append(nm)
    if req_missing:
        issues.append({"code":"missing_required_fields","msg":f"Missing: {', '.join(req_missing)}"})

    # ---- Section 6: Date rules
    today = context.get("today") or datetime.now(timezone.utc).date()
    inv_date = _parse_date(invoice_dt)
    checks["invoice_date_parsed"] = bool(inv_date)

    if inv_date:
        if inv_date > today:
            issues.append({"code":"date_in_future","msg":"Invoice date is in the future."})
        if inv_date < (today - timedelta(days=365)):
            issues.append({"code":"date_too_old","msg":"Invoice date is more than 1 year in the past."})

    # ---- Section 6: Duplicate invoice IDs
    known = context.get("known_invoice_ids")
    if isinstance(known, (set, list)) and invoice_id:
        if invoice_id in set(known):
            issues.append({"code":"duplicate_invoice_id","msg":f"Duplicate invoice id: {invoice_id}"})
        else:
            checks["duplicate_check"] = "not_found"
    else:
        checks["duplicate_check"] = None  # not evaluated

    # ---- Section 6: Line item total must match invoice total
    items = _collect_line_items(payload)
    computed["line_items_count"] = len(items)
    li_sum = _sum_line_items(items)
    computed["line_items_sum"] = li_sum

    subtotal = _to_float(_get_first(payload, "subtotal", "sub_total", "sub total"))
    tax      = _to_float(_get_first(payload, "tax", "tax_amount"))
    invoice_total = _to_float(total_amt)

    # Tolerate subtotal+tax ~= total if provided; else compare li_sum to total
    tolerance = 0.01
    if invoice_total is not None:
        if subtotal is not None or tax is not None:
            expected = (subtotal or 0.0) + (tax or 0.0)
            computed["expected_total_from_subtotals"] = expected
            if abs(expected - invoice_total) > tolerance:
                issues.append({"code":"total_mismatch",
                               "msg": f"Subtotal(+tax) {expected:.2f} != Total {invoice_total:.2f}"})
        # Compare line items to invoice total
        if items:
            if abs(li_sum - invoice_total) > tolerance:
                issues.append({"code":"line_items_sum_mismatch",
                               "msg": f"Sum(line items) {li_sum:.2f} != Total {invoice_total:.2f}"})
    else:
        issues.append({"code":"total_missing_or_non_numeric","msg":"Invoice total is missing or not numeric."})

    # ---- Section 6: Apply other rule systems (delegated)
    # These are gateways: we mark them as pending so other specific validators can run.
    for dep, code in [
        ("vendor management validation rules", "vendor_rules_pending"),
        ("purchase order validation rules",    "po_rules_pending"),
        ("statement of work validation rules", "sow_rules_pending"),
        ("contract validation rules",          "contract_rules_pending"),
    ]:
        issues.append({"code": code, "msg": f"Apply {dep} (external check)."})

    # ---- Outcome
    status = _status_from_issues(issues)
    return {
        "tool": "invoice_verification",
        "status": status,                  # 'valid' | 'needs_review' | 'rejected'
        "issues": issues,                  # list of dicts with 'code' and 'msg'
        "checks": checks,                  # what we evaluated
        "computed": computed,              # useful numbers for downstream tools
        "invoice_id": invoice_id,
        "vendor_name": vendor_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def run_invoice_verification(json_input: str):
    """Backwards-compatible entrypoint (string in, dict out)."""
    payload = json.loads(json_input) if isinstance(json_input, str) else json_input
    # You can weave context here if your pipeline sets it (e.g., global registry):
    context = {}
    # Example: if your extractor supplies 'source_format' in payload, we pick it up automatically in _run_invoice_verification
    return _run_invoice_verification(payload, context)
