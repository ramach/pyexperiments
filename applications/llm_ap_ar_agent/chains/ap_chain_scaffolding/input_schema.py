from pydantic import BaseModel, Field

class APInvoiceInput(BaseModel):
    vendor: str = Field(..., description="Vendor name")
    invoice_id: str = Field(..., description="Invoice ID")
    amount: float = Field(..., description="Invoice amount")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD")