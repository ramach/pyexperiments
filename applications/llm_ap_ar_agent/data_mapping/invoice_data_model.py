from pydantic import BaseModel
from typing import Optional

# Define the flexible Invoice data model
class InvoiceData(BaseModel):
    invoice_id: Optional[str] = None
    vendor: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    purchase_order: Optional[str] = None
    payment_method: Optional[str] = None
    extracted_text: Optional[str] = None
    # Add other fields as needed
