from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Optional

class BankTransferTopUpRequest(BaseModel):
    wallet_id: str
    amount: Decimal = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3, description="ISO currency code, e.g. USD")
    sender_name: Optional[str] = None
    sender_bank: Optional[str] = None
    reference: Optional[str] = None
    idempotency_key: Optional[str] = None

class BankTransferTopUpResponse(BaseModel):
    tx_id: str
    wallet_id: str
    status: str
    currency: str
    amount: Decimal
    net_amount: Decimal
    reference: Optional[str] = None
