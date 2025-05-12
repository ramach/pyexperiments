# agents/run_time_card_agent.py

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools.time_card_parser_tool import get_time_card_parser_tool

def run_time_card_agent(user_query: str, filepath: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    tool = get_time_card_parser_tool(filepath)

    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.run(user_query)
