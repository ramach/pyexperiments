# agents/llm_ap_ar_agent.py
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from .openai_function_handler import get_openai_tools


def create_financial_agent():
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    tools = get_openai_tools()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="openai-functions",
        verbose=True,
    )
    return agent


# Entry point for invoking queries via the agent
def run_agent_query(agent, query: str):
    return agent.run(query)
