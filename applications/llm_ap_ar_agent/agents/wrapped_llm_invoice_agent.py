from langchain.tools import Tool

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

class ContractData(BaseModel):
    client_name: Optional[str] = None
    consulting_firm: Optional[str] = None
    scope_of_work: Optional[str] = None
    fees: Optional[str] = None
    payment_terms: Optional[str] = None
    termination_clause: Optional[str] = None

from typing import Dict, Any
from pydantic import BaseModel

class LLMInvoiceAgentInput(BaseModel):
    query: str
    combined_data: Dict[str, Any]

class InvoiceAgentInput(BaseModel):
    query: str = Field(..., description="Natural language question or instruction for the invoice agent")
    invoice: Optional[Dict] = None
    purchase_order: Optional[Dict] = None
    contract: Optional[Dict] = None
    business_rules: Optional[List[Dict]] = None


def run_invoice_verification(invoice_data, business_rules):
    required_fields = ["invoice_id", "vendor_name", "invoice_date", "amount_due"]
    missing_fields = [f for f in required_fields if f not in invoice_data or not invoice_data[f]]

    compliance_issues = []
    if business_rules:
        max_allowed_amount = business_rules.get("max_invoice_amount")
        if max_allowed_amount and invoice_data.get("amount_due", 0) > max_allowed_amount:
            compliance_issues.append(f"Amount due exceeds allowed maximum of {max_allowed_amount}")

    result = {
        "status": "success" if not missing_fields and not compliance_issues else "failed",
        "missing_fields": missing_fields,
        "compliance_issues": compliance_issues
    }
    return result


def run_po_matching(invoice_data, purchase_order_data):
    mismatches = []
    if not invoice_data or not purchase_order_data:
        return {"status": "failed", "reason": "Missing invoice or purchase order data"}

    if invoice_data.get("po_number") != purchase_order_data.get("po_number"):
        mismatches.append("PO number does not match")
    if invoice_data.get("amount_due") != purchase_order_data.get("total_amount"):
        mismatches.append("Amount due does not match PO total amount")

    return {
        "status": "success" if not mismatches else "failed",
        "mismatches": mismatches
    }


def run_approval_process(invoice_data, contract_data, business_rules):
    issues = []
    if not invoice_data or not contract_data:
        return {"status": "failed", "reason": "Missing invoice or contract data"}

    if contract_data.get("client_name") not in invoice_data.get("client_details", ""):
        issues.append("Client name in invoice does not match contract")
    if business_rules and business_rules.get("require_contract_scope_check", False):
        if contract_data.get("scope_of_work") not in invoice_data.get("description", ""):
            issues.append("Scope of work mismatch between invoice and contract")

    return {
        "status": "approved" if not issues else "rejected",
        "approval_issues": issues
    }


def run_payment_processing(invoice_data, approval_result):
    if not invoice_data:
        return {"status": "failed", "reason": "Invoice data missing"}
    if not approval_result or approval_result.get("status") != "approved":
        return {"status": "failed", "reason": "Invoice not approved"}

    return {
        "status": "success",
        "payment_reference": f"PAY-{invoice_data.get('invoice_id', '0000')}"
    }


def run_llm_invoice_agent(query: str, combined_data: Dict) -> Dict:
    """
    Executes the invoice agent pipeline with optional sections and fallback defaults.
    Validates keys and provides logs for missing components.
    """
    from langchain.agents import initialize_agent, AgentType
    from langchain.chat_models import ChatOpenAI
    # Ensure top-level keys exist or use safe defaults
    invoice_data = combined_data.get("invoice", {})
    purchase_order_data = combined_data.get("purchase_order", {})
    contract_data = combined_data.get("contract", {})
    business_rules = combined_data.get("business_rules", [])
    # Log any missing parts
    if not invoice_data:
        logging.warning("⚠️ Missing 'invoice' data.")
    if not purchase_order_data:
        logging.warning("⚠️ Missing 'purchase_order' data.")
    if not contract_data:
        logging.warning("⚠️ Missing 'contract' data.")
    if not business_rules:
        logging.warning("⚠️ Missing 'business_rules'.")

    llm_invoice_agent_tool = Tool(
        name="LLMInvoiceAgent",
        func=run_llm_invoice_agent,
        description="Analyze invoice, PO, contract, and business rules using LLM",
        args_schema=LLMInvoiceAgentInput
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent_executor = initialize_agent(
        tools=[llm_invoice_agent_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    response = agent_executor.run({
        "query": query,
        "combined_data": combined_data
    })
    return eval(response)


def wrapped_llm_invoice_agent(input_data: InvoiceAgentInput) -> Dict:
    # Convert Pydantic model to plain dict
    combined_data = {
        "invoice": input_data.invoice or {},
        "purchase_order": input_data.purchase_order or {},
        "contract": input_data.contract  or {},
        "business_rules": input_data.business_rules or []
    }
    return run_llm_invoice_agent(input_data.query, combined_data)
