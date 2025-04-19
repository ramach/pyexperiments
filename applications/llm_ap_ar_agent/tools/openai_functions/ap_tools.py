import json
from chains.ap_chain_scaffolding.ap_chain import APChain
from chains.ap_chain_scaffolding.input_schema import APInvoiceInput

ap_chain = APChain()

def run_ap_chain_with_function_calling(input: str) -> str:
    try:
        payload = json.loads(input)
        validated = APInvoiceInput(**payload)
        return ap_chain.run(validated.dict())
    except Exception as e:
        return f"[APChain Error] {str(e)}"
