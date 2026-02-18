import os
import uuid
import requests

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"

def mask_account(acct: str) -> str:
    # Keep last 4 only
    last4 = acct[-4:]
    return f"****{last4}"\

def find_gl_id_by_name(name: str) -> int:
    base_url = os.getenv("FINERACT_BASE_URL", "http://host.docker.internal:8080/fineract-provider").rstrip("/")
    url = f"{base_url}/api/v1/glaccounts?limit=200"
    r = requests.get(url, headers=headers, auth=(user,pwd), timeout=20)
    r.raise_for_status()
    for a in r.json():
        if a["name"].upper() == name.upper():
            return int(a["id"])
    raise RuntimeError(f"GL account not found by name: {name}")

