from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .schemas_topup import BankTransferTopUpRequest, BankTransferTopUpResponse
from .services_topup import bank_transfer_topup

router = APIRouter(prefix="/v1/topups", tags=["topups"])

@router.post("/bank-transfer", response_model=BankTransferTopUpResponse)
def topup_bank_transfer(payload: BankTransferTopUpRequest, db: Session = Depends(get_db)):
    try:
        tx = bank_transfer_topup(
            db=db,
            wallet_id=payload.wallet_id,
            amount=payload.amount,
            currency=payload.currency,
            sender_name=payload.sender_name,
            sender_bank=payload.sender_bank,
            reference=payload.reference,
            idempotency_key=payload.idempotency_key,
        )
        return BankTransferTopUpResponse(
            tx_id=tx.tx_id,
            wallet_id=tx.wallet_id,
            status=tx.status.value if hasattr(tx.status, "value") else str(tx.status),
            currency=tx.currency,
            amount=tx.amount,
            net_amount=tx.net_amount,
            reference=tx.reference,
        )
    except ValueError as e:
        if str(e) == "WALLET_NOT_FOUND":
            raise HTTPException(status_code=404, detail="Wallet not found")
        raise HTTPException(status_code=400, detail=str(e))
