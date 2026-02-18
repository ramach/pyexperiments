from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from .db import get_db
from . import models
from typing import Optional

router = APIRouter(prefix="/v1/wallets", tags=["transactions"])

@router.get("/{wallet_id}/transactions")
def list_transactions(
        wallet_id: str,
        db: Session = Depends(get_db),
        currency: Optional[str] = Query(default=None),
        tx_type: Optional[str] = Query(default=None),
        status: Optional[str] = Query(default=None),
        q: Optional[str] = Query(default=None, description="search text"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
):
    stmt = select(models.WalletTransaction).where(models.WalletTransaction.wallet_id == wallet_id)

    if currency:
        stmt = stmt.where(models.WalletTransaction.currency == currency.upper())
    if tx_type:
        stmt = stmt.where(models.WalletTransaction.type == models.TxType[tx_type])
    if status:
        stmt = stmt.where(models.WalletTransaction.status == models.TxStatus[status])
    if q:
        like = f"%{q}%"
        stmt = stmt.where(models.WalletTransaction.searchable_text.ilike(like))

    stmt = stmt.order_by(desc(models.WalletTransaction.occurred_at)).limit(limit).offset(offset)
    rows = db.execute(stmt).scalars().all()

    return [
        {
            "tx_id": r.tx_id,
            "wallet_id": r.wallet_id,
            "type": r.type.name if hasattr(r.type, "name") else str(r.type),
            "status": r.status.name if hasattr(r.status, "name") else str(r.status),
            "currency": r.currency,
            "amount": str(r.amount),
            "fee": str(r.fee),
            "net_amount": str(r.net_amount),
            "reference": r.reference,
            "counterparty": r.counterparty,
            "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
        }
        for r in rows
    ]
