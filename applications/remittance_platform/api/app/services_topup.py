from __future__ import annotations

from decimal import Decimal
from datetime import date
import os

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from .utils import new_id


from . import models
from .fineract_ledger_service import post_fineract_journal_topup


def bank_transfer_topup(
        db: Session,
        wallet_id: str,
        amount: Decimal,
        currency: str,
        sender_name: str | None,
        sender_bank: str | None,
        reference: str | None,
        idempotency_key: str | None,
) -> models.WalletTransaction:
    # Basic wallet existence check
    wallet = db.get(models.Wallet, wallet_id)
    if not wallet:
        raise ValueError("WALLET_NOT_FOUND")

    cur = currency.upper()

    # Idempotency: if key provided, return existing tx if already processed
    if idempotency_key:
        existing = db.execute(
            select(models.WalletTransaction).where(
                models.WalletTransaction.wallet_id == wallet_id,
                models.WalletTransaction.reference == idempotency_key,  # reuse reference field for MVP
                models.WalletTransaction.type == models.TxType.TOPUP_BANK,
                )
        ).scalars().first()
        if existing:
            # If already completed, return immediately (idempotent replay)
            if existing.status == models.TxStatus.COMPLETED:
                return existing
            # If it previously failed, we'll attempt again (safe because we track ledger_posting)
            # If it is pending, also attempt again (common after restarts)

    # Ensure wallet_account exists for currency
    acct = db.execute(
        select(models.WalletAccount).where(
            models.WalletAccount.wallet_id == wallet_id,
            models.WalletAccount.currency == cur,
            )
    ).scalars().first()

    if not acct:
        acct = models.WalletAccount(
            wallet_account_id=new_id("acct"),
            wallet_id=wallet_id,
            currency=cur,
            status="ACTIVE",
        )
        db.add(acct)

    fee = Decimal("0.00")
    net = amount - fee

    searchable = " ".join([x for x in [sender_name, sender_bank, reference] if x])

    # Create local transaction first as PENDING (ledger will decide final state)

    tx = models.WalletTransaction(
        tx_id=new_id("tx"),
        wallet_id=wallet_id,
        type=models.TxType.TOPUP_BANK,
        status=models.TxStatus.PENDING,
        currency=cur,
        amount=amount,
        fee=fee,
        net_amount=net,
        reference=idempotency_key or reference,
        counterparty=sender_name,
        searchable_text=searchable or None,
    )
    db.add(tx)
    dedupe_ref = idempotency_key or reference  # single source of truth
    # 🔑 IMPORTANT: this flush can now fail due to uq_wallet_tx_reference
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        if dedupe_ref:
            existing = db.execute(
                select(models.WalletTransaction).where(
                    models.WalletTransaction.reference == dedupe_ref,
                    models.WalletTransaction.type == models.TxType.TOPUP_BANK,
                    )
            ).scalars().first()
            if existing:
                return existing
        raise

    # Create a ledger posting row (idempotency boundary on our side)
    posting = models.LedgerPosting(
        ledger_posting_id=new_id("lp"),
        wallet_tx_id=tx.tx_id,
        provider=models.LedgerProvider.FINERACT,
        status=models.LedgerPostingStatus.PENDING,
        #provider_ref=tx.tx_id, # could use idempotency_key if you prefer
    )
    db.add(posting)

    try:
        db.flush()
    except IntegrityError:
        # Unique(provider, wallet_tx_id) hit — someone already created it.
        db.rollback()

        # Load existing tx by idempotency key and return if completed
        if idempotency_key:
            existing = db.execute(
                select(models.WalletTransaction).where(
                    models.WalletTransaction.wallet_id == wallet_id,
                    models.WalletTransaction.reference == idempotency_key,
                    models.WalletTransaction.type == models.TxType.TOPUP_BANK,
                    )
            ).scalars().first()
            if existing:
                return existing
        raise

    # Fineract GL mapping (MVP: USD only)
    if cur != "USD":
        posting.status = models.LedgerPostingStatus.FAILED
        posting.last_error = f"Currency {cur} not configured for Fineract yet"
        tx.status = models.TxStatus.FAILED
        db.commit()
        raise ValueError("CURRENCY_NOT_CONFIGURED")

    settlement_gl_id = int(os.getenv("FINERACT_SETTLEMENT_GL_ID", "0"))
    wallet_liab_gl_id = int(os.getenv("FINERACT_WALLET_LIAB_GL_ID", "0"))
    if settlement_gl_id <= 0 or wallet_liab_gl_id <= 0:
        raise RuntimeError("Missing FINERACT GL IDs: set FINERACT_SETTLEMENT_GL_ID and FINERACT_WALLET_LIAB_GL_ID")


    try:
        provider_ref = post_fineract_journal_topup(
            wallet_tx_id=tx.tx_id,        # use tx_id as referenceNumber in Fineract
            currency=cur,
            amount=amount,
            transaction_date=date.today(),
            settlement_gl_id=settlement_gl_id,
            wallet_liab_gl_id=wallet_liab_gl_id,
        )

        posting.status = models.LedgerPostingStatus.POSTED
        posting.provider_ref = provider_ref
        tx.status = models.TxStatus.COMPLETED

        db.commit()
        db.refresh(tx)
        return tx

    except Exception as e:
        posting.status = models.LedgerPostingStatus.FAILED
        posting.last_error = str(e)
        tx.status = models.TxStatus.FAILED
        db.commit()
        raise
