import requests
from dataclasses import dataclass

@dataclass
class FineractAuth:
    tenant: str
    username: str
    password: str

class FineractClient:
    def __init__(self, base_url: str, auth: FineractAuth, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Fineract-Platform-TenantId": self.auth.tenant,
            "Accept": "application/json",
        }

    def create_journal_entry(self, payload: dict) -> dict:
        # Many deployments use POST /journalentries and a command param for "create"
        # Keep it configurable if needed.
        url = f"{self.base_url}/api/v1/journalentries"
        r = requests.post(
            url,
            headers=self._headers(),
            auth=(self.auth.username, self.auth.password),
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
