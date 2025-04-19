# demo/cli_chat_agent.py
import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agents.openai_function_handler import get_openai_tools

def main():
    load_dotenv()

    tools = get_openai_tools()  # Should be safely wrapped to avoid LLMChain instantiation during import

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    print("\n🤖 Ask anything about AP/AR or vendors (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = agent.run(query)
            print("Agent:", response)
        except Exception as e:
            print("[Error]", str(e))


if __name__ == "__main__":
    main()
