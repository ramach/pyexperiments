from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func
from . import models
from .utils import new_id

def ensure_wallet_account(db: Session, wallet_id: str, currency: str):
    acct = db.query(models.WalletAccount).filter_by(wallet_id=wallet_id, currency=currency).one_or_none()
    if acct:
        return acct
    acct = models.WalletAccount(wallet_account_id=new_id("wa"), wallet_id=wallet_id, currency=currency)
    db.add(acct)
    db.flush()
    return acct

def get_posted_balance(db: Session, wallet_id: str, currency: str) -> Decimal:
    # MVP "posted" balance = sum(COMPLETED tx net amounts, where topups are +, withdrawals are -
    # Store sign in net_amount already; we will set net_amount negative for withdrawals.
    total = db.query(func.coalesce(func.sum(models.WalletTransaction.net_amount), 0)).filter(
        models.WalletTransaction.wallet_id == wallet_id,
        models.WalletTransaction.currency == currency,
        models.WalletTransaction.status == models.TxStatus.COMPLETED
    ).scalar()
    return Decimal(total)

def get_reserved_balance(db: Session, wallet_id: str, currency: str) -> Decimal:
    total = db.query(func.coalesce(func.sum(models.WalletHold.amount), 0)).filter(
        models.WalletHold.wallet_id == wallet_id,
        models.WalletHold.currency == currency,
        models.WalletHold.status == models.HoldStatus.ACTIVE
    ).scalar()
    return Decimal(total)

def create_topup_completed(db: Session, wallet_id: str, currency: str, amount: Decimal, reference: str):
    ensure_wallet_account(db, wallet_id, currency)
    tx = models.WalletTransaction(
        tx_id=new_id("tx"),
        wallet_id=wallet_id,
        type=models.TxType.TOPUP_BANK,
        status=models.TxStatus.COMPLETED,
        currency=currency,
        amount=amount,
        fee=Decimal("0.00"),
        net_amount=amount,
        reference=reference,
        counterparty="BANK_TRANSFER",
        searchable_text=reference
    )
    db.add(tx)
    db.flush()
    return tx

def create_withdrawal(db: Session, wallet_id: str, bank_account_id: str, currency: str, amount: Decimal, idem: str | None):
    ensure_wallet_account(db, wallet_id, currency)

    posted = get_posted_balance(db, wallet_id, currency)
    reserved = get_reserved_balance(db, wallet_id, currency)
    available = posted - reserved
    if amount <= 0:
        raise ValueError("Amount must be > 0")
    if available < amount:
        raise ValueError("Insufficient available balance")

    w = models.Withdrawal(
        withdrawal_id=new_id("wd"),
        wallet_id=wallet_id,
        bank_account_id=bank_account_id,
        currency=currency,
        amount=amount,
        fee=Decimal("0.00"),
        status=models.WithdrawalStatus.SUBMITTED,
        idempotency_key=idem
    )
    db.add(w)
    db.flush()

    hold = models.WalletHold(
        hold_id=new_id("hld"),
        wallet_id=wallet_id,
        currency=currency,
        amount=amount,
        reason="WITHDRAWAL_PENDING",
        status=models.HoldStatus.ACTIVE,
        related_id=w.withdrawal_id,
        idempotency_key=idem
    )
    db.add(hold)
    db.flush()

    w.hold_id = hold.hold_id
    w.hub_reference = f"mockhub_{w.withdrawal_id}"
    db.flush()
    return w, hold

def finalize_withdrawal(db: Session, withdrawal_id: str, status: str, rail_tx_id: str):
    w = db.query(models.Withdrawal).filter_by(withdrawal_id=withdrawal_id).one()
    hold = db.query(models.WalletHold).filter_by(hold_id=w.hold_id).one()

    if status == "COMPLETED":
        w.status = models.WithdrawalStatus.COMPLETED
        hold.status = models.HoldStatus.CONSUMED

        # Record a completed withdrawal tx with NEGATIVE net_amount
        tx = models.WalletTransaction(
            tx_id=new_id("tx"),
            wallet_id=w.wallet_id,
            type=models.TxType.WITHDRAWAL,
            status=models.TxStatus.COMPLETED,
            currency=w.currency,
            amount=w.amount,
            fee=w.fee,
            net_amount=Decimal("0.00") - w.amount,  # negative
            reference=rail_tx_id,
            counterparty=f"BANK:{w.bank_account_id}",
            searchable_text=rail_tx_id
        )
        db.add(tx)
        db.flush()
        return w

    if status == "FAILED":
        w.status = models.WithdrawalStatus.FAILED
        hold.status = models.HoldStatus.RELEASED
        return w

    raise ValueError("Invalid status")
