import uuid

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"

def mask_account(acct: str) -> str:
    # Keep last 4 only
    last4 = acct[-4:]
    return f"****{last4}"
