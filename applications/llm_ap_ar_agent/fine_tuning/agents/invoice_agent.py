from langchain.agents import Tool, initialize_agent
from agents.tools import verify_invoice, check_business_rules

tools = [
    Tool(name="Verify Invoice", func=verify_invoice, description="Validates invoice fields"),
    Tool(name="Check Business Rules", func=check_business_rules, description="Checks rule compliance")
]

def get_invoice_agent(llm):
    return initialize_agent(tools, llm=llm, agent_type="zero-shot-react-description")

