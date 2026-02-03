from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from decimal import Decimal
from datetime import datetime

class WalletCreate(BaseModel):
    customer_id: str
    default_currency: str = "USD"

class WalletOut(BaseModel):
    wallet_id: str
    customer_id: str
    default_currency: str
    status: str
    created_at: datetime

class BalanceLine(BaseModel):
    currency: str
    posted: Decimal
    reserved: Decimal
    available: Decimal

class BalancesOut(BaseModel):
    wallet_id: str
    balances: List[BalanceLine]

class BankAccountCreate(BaseModel):
    country: str
    currency: str
    # MVP: accept plain string but store masked
    account_number: str = Field(min_length=4)
    routing_number: Optional[str] = None
    account_holder_name: Optional[str] = None

class BankAccountOut(BaseModel):
    bank_account_id: str
    wallet_id: str
    country: str
    currency: str
    masked_details: str
    status: str
    created_at: datetime

class WithdrawalCreate(BaseModel):
    wallet_id: str
    bank_account_id: str
    currency: str
    amount: Decimal

class WithdrawalOut(BaseModel):
    withdrawal_id: str
    wallet_id: str
    bank_account_id: str
    currency: str
    amount: Decimal
    fee: Decimal
    status: str
    created_at: datetime
    updated_at: datetime

class InboundCredit(BaseModel):
    reference: str  # e.g. WLT-<wallet_id>
    wallet_id: str
    currency: str
    amount: Decimal
    rail_tx_id: str

class PayoutCallback(BaseModel):
    withdrawal_id: str
    status: Literal["COMPLETED", "FAILED"]
    rail_tx_id: str

class TxOut(BaseModel):
    tx_id: str
    wallet_id: str
    type: str
    status: str
    currency: str
    amount: Decimal
    fee: Decimal
    net_amount: Decimal
    reference: Optional[str]
    counterparty: Optional[str]
    occurred_at: datetime

class TxListOut(BaseModel):
    items: List[TxOut]
