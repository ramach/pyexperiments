from langchain.tools import Tool
from tools.openai_functions.ap_tools import run_ap_chain_with_function_calling
from tools.openai_functions.ar_tools import run_ar_chain_with_function_calling

def get_openai_tools():
    return [
        Tool(
            name="ProcessAccountsPayable",
            func=run_ap_chain_with_function_calling,
            description="Analyze a vendor invoice for Accounts Payable using vendor name, amount, due date, and invoice ID."
        ),
        Tool(
            name="ProcessAccountsReceivable",
            func=run_ar_chain_with_function_calling,
            description="Analyze a customer invoice for Accounts Receivable using customer name, amount, due date, and invoice ID."
        )
    ]