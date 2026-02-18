from decimal import Decimal
from datetime import date
from app.ledger.config import LedgerSettings
from app.ledger.fineract_client import FineractClient, FineractAuth

class LedgerService:
    def __init__(self, settings: LedgerSettings):
        self.settings = settings
        self.client = FineractClient(
            base_url=settings.FINERACT_BASE_URL,
            auth=FineractAuth(
                tenant=settings.FINERACT_TENANT,
                username=settings.FINERACT_USER,
                password=settings.FINERACT_PASSWORD,
            ),
        )

    def post_bank_topup(
            self,
            *,
            wallet_tx_id: str,
            currency: str,
            amount: Decimal,
            transaction_date: date,
            gl_settlement_id: int,
            gl_wallet_liab_id: int,
            note: str | None = None,
    ) -> dict:
        # Fineract journal entry payload shape varies slightly by version.
        # The core idea: officeId, currencyCode, transactionDate, debits, credits.
        payload = {
            "officeId": self.settings.FINERACT_OFFICE_ID,
            "locale": "en",
            "dateFormat": "yyyy-MM-dd",
            "transactionDate": transaction_date.strftime("%Y-%m-%d"),
            "currencyCode": currency.upper(),
            "referenceNumber": wallet_tx_id,
            "comments": f"Wallet top-up {wallet_tx_id}",
            "debits": [{"glAccountId": gl_settlement_id, "amount": float(amount)}],
            "credits": [{"glAccountId": gl_wallet_liab_id, "amount": float(amount)}],
        }

        return self.client.create_journal_entry(payload)
