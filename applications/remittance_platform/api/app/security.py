from fastapi import Header, HTTPException
from .config import settings

def require_internal_api_key(x_internal_api_key: str = Header(default="")):
    if x_internal_api_key != settings.api_key_internal:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
