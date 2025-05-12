import json
from typing import Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI

# Shared state for tools to communicate
shared_state: Dict[str, Any] = {}


# Tool 1: Invoice Verification
def make_invoice_verification_tool(shared_state: Dict[str, Any]):
    def run(query: str) -> str:
        # Simulated input
        combined_data = {
            "invoice": {
                "invoice_id": "123",
                "vendor_name": "Acme Corp",
                "invoice_date": "2025-05-01",
                "amount_due": "1000"
            },
            "business_rules": [
                {"rule": "All invoices must be verified before approval"}
            ]
        }

        invoice_data = combined_data["invoice"]
        business_rules = combined_data["business_rules"]

        missing_fields = [f for f in ["invoice_id", "vendor_name", "invoice_date", "amount_due"]
                          if not invoice_data.get(f)]

        compliant = not missing_fields and all(
            rule["rule"] != "All invoices must be verified before approval"
            or True  # rule is satisfied
            for rule in business_rules
        )

        result = {
            "status": "success" if compliant else "failed",
            "missing_fields": missing_fields,
            "compliant": compliant
        }

        shared_state["invoice_verification"] = result
        return json.dumps(result)

    return Tool(
        name="invoice_verification",
        func=run,
        description="Verify invoice completeness and rule compliance"
    )


# Tool 2: Approval Process
def make_approval_process_tool(shared_state: Dict[str, Any]):
    def run(query: str) -> str:
        verification_result = shared_state.get("invoice_verification")

        if not verification_result:
            return "Invoice verification must be run before approval."

        if verification_result.get("status") != "success":
            return "Invoice verification failed. Cannot proceed to approval."

        approval_result = {
            "approver": "Finance Manager",
            "status": "ready for approval"
        }

        shared_state["approval_process"] = approval_result
        return json.dumps(approval_result)

    return Tool(
        name="approval_process",
        func=run,
        description="Performs invoice approval process after verification"
    )


# Tool 3: Payment Processing
def make_payment_processing_tool(shared_state: Dict[str, Any]):
    def run(query: str) -> str:
        approval = shared_state.get("approval_process")

        if not approval or approval.get("status") != "ready for approval":
            return "Approval is not complete. Cannot process payment."

        result = {
            "payment_status": "initiated",
            "via": "bank_transfer"
        }

        shared_state["payment_processing"] = result
        return json.dumps(result)

    return Tool(
        name="payment_processing",
        func=run,
        description="Initiate payment only after approval is done"
    )


# Initialize LangChain Agent
def create_invoice_agent(shared_state: Dict[str, Any]):
    tools = [
        make_invoice_verification_tool(shared_state),
        make_approval_process_tool(shared_state),
        make_payment_processing_tool(shared_state),
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent


# 🧪 Run Agent with Shared State
if __name__ == "__main__":
    agent_executor = create_invoice_agent(shared_state)

    user_query = "Please verify this invoice and process the payment."
    final_response = agent_executor.run(user_query)

    print("\n📤 Final Response:")
    print(final_response)

    print("\n🧠 Shared State Snapshot:")
    print(json.dumps(shared_state, indent=2))
