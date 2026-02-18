from __future__ import annotations

import os
from decimal import Decimal
import requests
from datetime import date

def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var {name}")
    return v

def post_fineract_journal_topup(
        *,
        wallet_tx_id: str,
        currency: str,
        amount: Decimal,
        transaction_date: date,
        settlement_gl_id: int,
        wallet_liab_gl_id: int,
) -> str:
    """
    Returns Fineract transactionId
    """
    #base_url = _env("FINERACT_BASE_URL")
    base_url = os.getenv("FINERACT_BASE_URL", "http://host.docker.internal:8080/fineract-provider").rstrip("/")
    tenant = _env("FINERACT_TENANT", "default")
    user = _env("FINERACT_USER")
    pwd = _env("FINERACT_PASSWORD")
    office_id = int(_env("FINERACT_OFFICE_ID", "1"))

    payload = {
        "officeId": office_id,
        "locale": "en",
        "dateFormat": "dd MMMM yyyy",
        "transactionDate": transaction_date.strftime("%d %B %Y"),
        "currencyCode": currency.upper(),
        "referenceNumber": wallet_tx_id,
        "comments": f"Wallet bank top-up {wallet_tx_id}",
        "debits": [{"glAccountId": settlement_gl_id, "amount": float(amount)}],
        "credits": [{"glAccountId": wallet_liab_gl_id, "amount": float(amount)}],
    }
    #url = "http://fineract:8080/fineract-provider/api/v1/journalentries"

    url = f"{base_url}/api/v1/journalentries"   # ✅ correct
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Fineract-Platform-TenantId": tenant,
    }
    print("FINERACT base_url =", base_url, flush=True)
    print("FINERACT post url =", url, flush=True)
    print("FINERACT tenant =", tenant, "user =", user, "office_id =", office_id, flush=True)

    r = requests.post(
        url,
        headers=headers,
        auth=(user, pwd),
        json=payload,
        timeout=30,
    )
    print("FINERACT status =", r.status_code, flush=True)
    print("FINERACT response headers content-type =", r.headers.get("content-type"), flush=True)
    print("FINERACT response text =", r.text[:2000], flush=True)  # cap to avoid log spam
    # If non-2xx, raise with body visible in logs
    r.raise_for_status()
    data = r.json() if r.text else {}
    txid = data.get("transactionId")
    if not txid:
        raise RuntimeError(f"Unexpected Fineract response: {data}")
    return txid
