import json

from langchain.tools import Tool, StructuredTool

from pydantic.v1 import BaseModel
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from typing import Dict, List, Optional, Any
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ContractData(BaseModel):
    client_name: Optional[str] = None
    consulting_firm: Optional[str] = None
    scope_of_work: Optional[str] = None
    fees: Optional[str] = None
    payment_terms: Optional[str] = None
    termination_clause: Optional[str] = None

class LLMInvoiceAgentInput(BaseModel):
    input: str
    input_data: Dict[str, Any]

def run_invoice_verification(input_data: Dict[str, Any]):
    invoice_data = input_data.get("invoice", {})
    invoice_fields = json.dumps(invoice_data)
    logger.debug(f"[run_invoice_verification], {invoice_data}")
    logger.debug(f"[run_invoice_verification], {invoice_fields}")
    business_rules = input_data.get("business_rules", [])
    required_fields = ["invoice_id", "vendor", "date", "amount"]
    missing_fields = [f for f in required_fields if f not in invoice_data or not invoice_data[f]]

    compliance_issues = []
    if business_rules:
        invoice_business_rule = business_rules[0]
        invoice_business_rule_fields = json.dumps(invoice_business_rule)
        logger.debug(f"[run_invoice_verification], {invoice_business_rule}")
        logger.debug(f"[run_invoice_verification], {invoice_business_rule_fields}")
        if "must be verified" in invoice_business_rule_fields:
            if missing_fields:
                compliance_issues.append("compliance issues in invoice identified")
    result = {
        "status": "success" if not missing_fields and not compliance_issues else "failed",
        "missing_fields": missing_fields,
        "compliance_issues": compliance_issues
    }
    return result


def run_po_matching(input_data: Dict[str, Any]):
    invoice_data = input_data.get("invoice", {})
    purchase_order_data = input_data.get("purchase_order", {})
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


def run_approval_process(input_data: Dict[str, Any]):
    invoice_data = input_data.get("invoice", {})
    business_rules = input_data.get("business_rules", [])
    contract_data = input_data.get("contract", {})
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


def run_payment_processing(input_data: Dict[str, Any]):
    invoice_data = input_data.get("invoice", {})
    approval_result = input_data.get("approval", {})
    if not invoice_data:
        return {"status": "failed", "reason": "Invoice data missing"}
    if not approval_result or approval_result.get("status") != "approved":
        return {"status": "failed", "reason": "Invoice not approved"}

    return {
        "status": "success",
        "payment_reference": f"PAY-{invoice_data.get('invoice_id', '0000')}"
    }


def build_invoice_agent_tools(input_data: Dict[str, Any]):
    def make_tool_runner(tool_logic_func):
        def tool_runner(_: str) -> str:
            return tool_logic_func(input_data)
        return tool_runner

    tools = [
        Tool(
            name="invoice_verification",
            func=make_tool_runner(run_invoice_verification),
            description="Verify invoice details (e.g., ID, vendor, amount)"
        ),
        Tool(
            name="po_matching",
            func=make_tool_runner(run_po_matching),
            description="Match the invoice with the correct PO"
        ),
        Tool(
            name="approval_process",
            func=make_tool_runner(run_approval_process),
            description="Determine who needs to approve the invoice"
        ),
        Tool(
            name="payment_processing",
            func=make_tool_runner(run_payment_processing),
            description="Simulate invoice payment process"
        )
    ]
    return tools


def run_llm_invoice_agent(query: str, input_data: Dict[str, Any]) -> str:
    tools = build_invoice_agent_tools(input_data)
    agent_executor = initialize_agent(
        tools=tools,
        llm = ChatOpenAI(temperature=0, model_name="gpt-4"),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent_executor.run(query)

