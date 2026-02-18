from fastapi import FastAPI, Depends, Header, HTTPException, Response
from sqlalchemy.orm import Session
from typing import Optional
import csv
import io
import os

from dotenv import load_dotenv
load_dotenv()

from .db import get_db
from . import models, schemas
from .utils import new_id, mask_account
from .security import require_internal_api_key

from .routes_transactions import router as tx_router
from .routes_topups_api import router as topups_router

app = FastAPI(title="Remit Wallet MVP API")
app.include_router(tx_router)
app.include_router(topups_router)

from .services import (
    ensure_wallet_account, get_posted_balance, get_reserved_balance,
    create_topup_completed, create_withdrawal, finalize_withdrawal
)

app = FastAPI(title="Remit Wallet MVP API")

# -------- Slice 1: create wallet + balances + tx list --------

@app.post("/v1/wallets", response_model=schemas.WalletOut)
def create_wallet(payload: schemas.WalletCreate, db: Session = Depends(get_db)):
    w = models.Wallet(
        wallet_id=new_id("wlt"),
        customer_id=payload.customer_id,
        default_currency=payload.default_currency,
        status=models.WalletStatus.ACTIVE
    )
    db.add(w)
    db.flush()
    # Create default currency account
    ensure_wallet_account(db, w.wallet_id, w.default_currency)
    db.commit()
    db.refresh(w)
    return w

@app.get("/v1/wallets/{wallet_id}/balances", response_model=schemas.BalancesOut)
def get_balances(wallet_id: str, db: Session = Depends(get_db)):
    wallet = db.query(models.Wallet).filter_by(wallet_id=wallet_id).one_or_none()
    if not wallet:
        raise HTTPException(404, "Wallet not found")

    accounts = db.query(models.WalletAccount).filter_by(wallet_id=wallet_id).all()
    lines = []
    for a in accounts:
        posted = get_posted_balance(db, wallet_id, a.currency)
        reserved = get_reserved_balance(db, wallet_id, a.currency)
        available = posted - reserved
        lines.append(schemas.BalanceLine(currency=a.currency, posted=posted, reserved=reserved, available=available))
    return schemas.BalancesOut(wallet_id=wallet_id, balances=lines)

@app.get("/v1/wallets/{wallet_id}/transactions", response_model=schemas.TxListOut)
def list_transactions(
        wallet_id: str,
        currency: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        db: Session = Depends(get_db)
):
    q = db.query(models.WalletTransaction).filter(models.WalletTransaction.wallet_id == wallet_id)
    if currency:
        q = q.filter(models.WalletTransaction.currency == currency)
    if type:
        q = q.filter(models.WalletTransaction.type == models.TxType(type))
    if status:
        q = q.filter(models.WalletTransaction.status == models.TxStatus(status))
    items = q.order_by(models.WalletTransaction.occurred_at.desc()).limit(min(limit, 200)).all()
    return {"items": items}

# -------- Bank account linking --------

@app.post("/v1/wallets/{wallet_id}/bank-accounts", response_model=schemas.BankAccountOut)
def add_bank_account(wallet_id: str, payload: schemas.BankAccountCreate, db: Session = Depends(get_db)):
    wallet = db.query(models.Wallet).filter_by(wallet_id=wallet_id).one_or_none()
    if not wallet:
        raise HTTPException(404, "Wallet not found")

    ba = models.BankAccount(
        bank_account_id=new_id("ba"),
        wallet_id=wallet_id,
        country=payload.country,
        currency=payload.currency,
        masked_details=mask_account(payload.account_number),
        status="ACTIVE"
    )
    db.add(ba)
    db.commit()
    db.refresh(ba)
    return ba

@app.get("/v1/wallets/{wallet_id}/bank-accounts", response_model=list[schemas.BankAccountOut])
def list_bank_accounts(wallet_id: str, db: Session = Depends(get_db)):
    return db.query(models.BankAccount).filter_by(wallet_id=wallet_id).order_by(models.BankAccount.created_at.desc()).all()

# -------- Slice 3: Bank transfer topup instructions + internal ingest --------

@app.get("/v1/topups/bank/instructions")
def bank_topup_instructions(wallet_id: str, currency: str = "USD"):
    # In real life: virtual account/reference; MVP: use wallet_id as reference
    return {
        "method": "BANK_TRANSFER",
        "currency": currency,
        "beneficiary_name": "MVP Wallet Co",
        "bank_name": "Partner Bank (mock)",
        "account": "000111222",
        "routing": "000000000",
        "reference": f"WLT-{wallet_id}",
        "note": "Include reference to credit your wallet."
    }

from .routes_topup import bank_transfer_topup

from sqlalchemy.exc import IntegrityError

@app.post("/internal/inbound-credits", dependencies=[Depends(require_internal_api_key)])
def ingest_inbound_credit(payload: schemas.InboundCredit, db: Session = Depends(get_db)):
    wallet = db.query(models.Wallet).filter_by(wallet_id=payload.wallet_id).one_or_none()
    if not wallet:
        raise HTTPException(404, "Wallet not found")

    # optional: quick pre-check (fast path)
    existing = db.query(models.WalletTransaction).filter_by(reference=payload.rail_tx_id).one_or_none()
    if existing:
        return {"status": "DUPLICATE_IGNORED", "tx_id": existing.tx_id, "tx_status": existing.status.value}

    try:
        tx = bank_transfer_topup(
            db=db,
            wallet_id=payload.wallet_id,
            amount=payload.amount,
            currency=payload.currency,
            sender_name=None,
            sender_bank=None,
            reference=payload.reference,
            idempotency_key=payload.rail_tx_id,   # ensure you pass the same value used as reference
        )
        db.commit()
        return {"status": "OK", "tx_id": tx.tx_id, "tx_status": tx.status.value}

    except IntegrityError:
        # someone already inserted the same reference (race/retry)
        db.rollback()
        existing = db.query(models.WalletTransaction).filter_by(reference=payload.rail_tx_id).one_or_none()
        if existing:
            return {"status": "DUPLICATE_IGNORED", "tx_id": existing.tx_id, "tx_status": existing.status.value}
        # if we still can't find it, re-raise so you notice
        raise

@app.post("/internal/inbound-credits_old", dependencies=[Depends(require_internal_api_key)])
def ingest_inbound_credit_old(payload: schemas.InboundCredit, db: Session = Depends(get_db)):
    wallet = db.query(models.Wallet).filter_by(wallet_id=payload.wallet_id).one_or_none()
    if not wallet:
        raise HTTPException(404, "Wallet not found")

    try:
        tx = bank_transfer_topup(
            db=db,
            wallet_id=payload.wallet_id,
            amount=payload.amount,
            currency=payload.currency,
            sender_name=None,
            sender_bank=None,
            reference=payload.rail_tx_id,
            idempotency_key=payload.rail_tx_id,
        )
        return {"status": "OK", "tx_id": tx.tx_id, "tx_status": tx.status.value}
    except ValueError as e:
        raise HTTPException(400, str(e))

# -------- Slice 4: Withdrawals + internal callback --------

@app.post("/v1/withdrawals", response_model=schemas.WithdrawalOut)
def create_withdrawal_api(
        payload: schemas.WithdrawalCreate,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        db: Session = Depends(get_db)
):
    # Idempotency: if same wallet + idem exists, return existing
    if idempotency_key:
        existing = db.query(models.Withdrawal).filter_by(wallet_id=payload.wallet_id, idempotency_key=idempotency_key).one_or_none()
        if existing:
            return existing

    try:
        w, _hold = create_withdrawal(
            db,
            wallet_id=payload.wallet_id,
            bank_account_id=payload.bank_account_id,
            currency=payload.currency,
            amount=payload.amount,
            idem=idempotency_key
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    db.commit()
    db.refresh(w)
    return w

@app.post("/internal/payout-callback", dependencies=[Depends(require_internal_api_key)])
def payout_callback(payload: schemas.PayoutCallback, db: Session = Depends(get_db)):
    try:
        w = finalize_withdrawal(db, payload.withdrawal_id, payload.status, payload.rail_tx_id)
    except Exception as e:
        raise HTTPException(400, str(e))
    db.commit()
    return {"status": "OK", "withdrawal_status": w.status}

# -------- Slice 5: CSV statement export (simple synchronous) --------

@app.get("/v1/wallets/{wallet_id}/statement.csv")
def statement_csv(wallet_id: str, currency: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(models.WalletTransaction).filter(models.WalletTransaction.wallet_id == wallet_id)
    if currency:
        q = q.filter(models.WalletTransaction.currency == currency)
    items = q.order_by(models.WalletTransaction.occurred_at.asc()).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["occurred_at", "type", "status", "currency", "amount", "fee", "net_amount", "reference", "counterparty"])
    for t in items:
        writer.writerow([t.occurred_at.isoformat(), t.type.value, t.status.value, t.currency, str(t.amount), str(t.fee), str(t.net_amount), t.reference or "", t.counterparty or ""])

    data = buf.getvalue().encode("utf-8")
    return Response(content=data, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{wallet_id}_statement.csv"'
    })
