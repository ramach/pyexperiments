import json
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from tools.invoice_tools_with_business_rules import (
    invoice_verification_tool,
    po_validation_tool,
    approval_process_tool,
    payment_processing_tool
)

llm = ChatOpenAI(temperature=0, model="gpt-4")

# All available tools
TOOLS = [
    invoice_verification_tool,
    po_validation_tool,
    approval_process_tool,
    payment_processing_tool,
]

# Initialize agent with all tools
agent = initialize_agent(
    TOOLS,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

def run_llm_invoice_agent(query: str, input_data: Dict) -> Dict:
    import json
    """
    Executes one or more invoice-processing tools depending on query.
    Runs all tools in sequence for generic query; otherwise invokes the most relevant.
    """
    try:
        results = {}

        # Determine intent from the query
        query_lower = query.lower().strip()
        run_all = query_lower in [
            "analyze this invoice", "process this", "run all", "full validation"
        ]

        if run_all:
            # Run tools sequentially and collect outputs
            for tool in TOOLS:
                tool_name = tool.name
                try:
                    output = tool.run(input_data)
                    results[tool_name] = {
                        "result": output,
                        "confidence": "95%"  # Placeholder or optionally from LLM scoring
                    }
                except Exception as e:
                    results[tool_name] = {"error": str(e)}
        else:
            # Use agent to determine appropriate tool
            tool_result = agent.run(input=input_data, query=query)
            results["agent_response"] = {
                "result": tool_result,
                "confidence": "90%"  # Optional score if scoring available
            }

        return results

    except Exception as e:
        return {"error": str(e)}
