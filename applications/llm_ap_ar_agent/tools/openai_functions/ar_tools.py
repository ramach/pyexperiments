import json
from chains.ar_chain_scaffolding.ar_chain import ARChain
from pydantic import BaseModel

ar_chain = ARChain()

class ARInvoiceInput(BaseModel):
    customer: str
    invoice_id: str
    amount: float
    due_date: str

def run_ar_chain_with_function_calling(input: str) -> str:
    try:
        payload = json.loads(input)
        validated = ARInvoiceInput(**payload)
        return ar_chain.run(validated.dict())
    except Exception as e:
        return f"[APChain Error] {str(e)}"