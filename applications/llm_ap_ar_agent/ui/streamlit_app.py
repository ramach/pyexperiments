# ui/streamlit_app.py
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from ..agents.openai_function_handler import get_openai_tools
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="AP/AR Chat Agent", page_icon="🤖")
st.title("💼 Accounts Payable / Receivable Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LLM agent
@st.cache_resource
def get_agent():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = get_openai_tools()
    return initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

agent = get_agent()

# Input UI
user_input = st.chat_input("Ask something about AP, AR, or vendors...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(user_input)
            st.session_state.chat_history.append((user_input, response))
        except Exception as e:
            response = f"[Error] {str(e)}"
            st.session_state.chat_history.append((user_input, response))

# Display chat history
for user_q, agent_a in reversed(st.session_state.chat_history):
    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(agent_a)