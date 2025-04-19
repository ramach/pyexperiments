from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.schema import HumanMessage
from agents.openai_function_handler import openai_functions, get_tool_response
from retrievers.query_optimizer import QueryOptimizer


def query_financial_assistant(query, mode, qa_chain_ap, qa_chain_ar, query_optimizer_ap, query_optimizer_ar, agent):
    if mode == "ap":
        context = query_optimizer_ap.optimize_query(query)
        return agent.run(input=context)
    elif mode == "ar":
        context = query_optimizer_ar.optimize_query(query)
        return agent.run(input=context)
    elif mode == "vendor":
        return agent.run(input=query)
    elif mode == "public":
        return agent.run(input=query)
    else:
        return "Invalid mode. Choose from: ap, ar, vendor, public."


def get_agent_with_function_calling():
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    tools = [
        Tool(
            name=fn["name"],
            func=lambda x, f=fn["function"]: get_tool_response(f, x),
            description=fn["description"]
        ) for fn in openai_functions
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="openai-functions",
        verbose=True
    )
