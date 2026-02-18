from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from .db import get_db
from . import schemas
from .security import require_internal_api_key
from .routes_topup import bank_transfer_topup  # your service function

router = APIRouter(prefix="", tags=["topups"])

@router.post("/internal/inbound-credits", dependencies=[Depends(require_internal_api_key)])
def ingest_inbound_credit(
        payload: schemas.InboundCredit,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        db: Session = Depends(get_db)
):
    try:
        tx = bank_transfer_topup(
            db=db,
            wallet_id=payload.wallet_id,
            amount=payload.amount,
            currency=payload.currency,
            sender_name=payload.sender_name if hasattr(payload, "sender_name") else None,
            sender_bank=payload.sender_bank if hasattr(payload, "sender_bank") else None,
            reference=payload.rail_tx_id,
            idempotency_key=idempotency_key or payload.rail_tx_id,
        )
        return {"status": "OK", "tx_id": tx.tx_id, "tx_status": tx.status.value}
    except ValueError as e:
        raise HTTPException(400, str(e))
