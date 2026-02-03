import enum
from sqlalchemy import (
    Column, String, DateTime, Numeric, Text, ForeignKey, Enum, func, Index
)
from sqlalchemy.orm import relationship
from .db import Base

class WalletStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"

class HoldStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    CONSUMED = "CONSUMED"

class TxStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"

class TxType(str, enum.Enum):
    TOPUP_BANK = "TOPUP_BANK"
    WITHDRAWAL = "WITHDRAWAL"
    FEE = "FEE"
    ADJUSTMENT = "ADJUSTMENT"

class WithdrawalStatus(str, enum.Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REVERSED = "REVERSED"
    MANUAL_REVIEW = "MANUAL_REVIEW"

class Wallet(Base):
    __tablename__ = "wallet"
    wallet_id = Column(String, primary_key=True)
    customer_id = Column(String, unique=True, nullable=False)
    default_currency = Column(String, nullable=False, default="USD")
    status = Column(Enum(WalletStatus), nullable=False, default=WalletStatus.ACTIVE)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    accounts = relationship("WalletAccount", back_populates="wallet", cascade="all,delete-orphan")

class WalletAccount(Base):
    __tablename__ = "wallet_account"
    wallet_account_id = Column(String, primary_key=True)
    wallet_id = Column(String, ForeignKey("wallet.wallet_id"), nullable=False)
    currency = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ACTIVE")

    wallet = relationship("Wallet", back_populates="accounts")

    __table_args__ = (Index("uq_wallet_currency", "wallet_id", "currency", unique=True),)

class WalletHold(Base):
    __tablename__ = "wallet_hold"
    hold_id = Column(String, primary_key=True)
    wallet_id = Column(String, ForeignKey("wallet.wallet_id"), nullable=False)
    currency = Column(String, nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    reason = Column(String, nullable=False)  # WITHDRAWAL_PENDING, etc.
    status = Column(Enum(HoldStatus), nullable=False, default=HoldStatus.ACTIVE)
    idempotency_key = Column(String, nullable=True)
    related_id = Column(String, nullable=True)  # withdrawal_id etc
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    released_at = Column(DateTime(timezone=True), nullable=True)

class WalletTransaction(Base):
    __tablename__ = "wallet_transaction"
    tx_id = Column(String, primary_key=True)
    wallet_id = Column(String, ForeignKey("wallet.wallet_id"), nullable=False)
    type = Column(Enum(TxType), nullable=False)
    status = Column(Enum(TxStatus), nullable=False)
    currency = Column(String, nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    fee = Column(Numeric(18, 2), nullable=False, default=0)
    net_amount = Column(Numeric(18, 2), nullable=False)
    reference = Column(String, nullable=True)
    counterparty = Column(String, nullable=True)
    searchable_text = Column(Text, nullable=True)
    occurred_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_tx_wallet_time", "wallet_id", "occurred_at"),
        Index("idx_tx_filters", "wallet_id", "currency", "type", "status", "occurred_at"),
    )

class BankAccount(Base):
    __tablename__ = "bank_account"
    bank_account_id = Column(String, primary_key=True)
    wallet_id = Column(String, ForeignKey("wallet.wallet_id"), nullable=False)
    country = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    masked_details = Column(String, nullable=False)  # store masked only in MVP
    status = Column(String, nullable=False, default="ACTIVE")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Withdrawal(Base):
    __tablename__ = "withdrawal"
    withdrawal_id = Column(String, primary_key=True)
    wallet_id = Column(String, ForeignKey("wallet.wallet_id"), nullable=False)
    bank_account_id = Column(String, ForeignKey("bank_account.bank_account_id"), nullable=False)
    currency = Column(String, nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    fee = Column(Numeric(18, 2), nullable=False, default=0)
    status = Column(Enum(WithdrawalStatus), nullable=False, default=WithdrawalStatus.CREATED)
    hub_reference = Column(String, nullable=True)
    hold_id = Column(String, nullable=True)
    idempotency_key = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_withdrawal_wallet_time", "wallet_id", "created_at"),
    )
