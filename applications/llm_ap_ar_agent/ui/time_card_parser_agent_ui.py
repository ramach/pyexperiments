
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from utils.text_extraction import extract_timecard_from_excel
import pandas as pd
import json
from io import BytesIO
from datetime import datetime

from utils.text_extraction import extract_timecard_metadata_generic
from utils.text_extraction import robust_extract_text
from agents.llm_invoice_agent import  map_extracted_text_to_timecard_data_with_confidence_score
# === STEP 1: Time Card Parser ===

def parse_time_card_excel(file: BytesIO) -> dict:
    df = pd.read_excel(file, header=None)
    result = {}

    def find_value(label_keywords):
        for i in range(len(df)):
            for j in range(len(df.columns)):
                cell = str(df.iat[i, j]).strip().lower()
                if any(kw in cell for kw in label_keywords):
                    return str(df.iat[i, j + 1]) if j + 1 < len(df.columns) else None
        return None

    result["employee_name"] = find_value(["employee", "name"]) or "Unknown"
    result["manager_name"] = find_value(["manager"]) or "Unknown"

    try:
        total_hours = float(find_value(["Total hours", "total hrs"]) or 0)
        result["total_hours"] = total_hours
    except:
        result["total_hours"] = 0.0

    try:
        hourly_rate = float(find_value(["Rate per hour", "rate"]) or 0)
        result["hourly_rate"] = hourly_rate
    except:
        result["hourly_rate"] = 0.0

    result["total_amount"] = round(result["total_hours"] * result["hourly_rate"], 2)
    result["rule_compliance"] = (
        "Compliant" if result["total_hours"] >= 40 else "Non-compliant: Less than 40 hours"
    )

    return result

# === STEP 2: LangChain Tool Wrapper ===

def get_time_card_tool(parsed_data: dict) -> Tool:
    def tool_func(_input: str) -> str:
        return json.dumps(parsed_data, indent=2, default=str)

    return Tool(
        name="TimeCardExtractor",
        func=tool_func,
        description="Use this tool to answer questions about the employee's time card, work hours, and payroll compliance."
    )


# === STEP 3: Agent Runner ===

def run_time_card_agent(user_query: str, parsed_data: dict) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    tool = get_time_card_tool(parsed_data)

    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.run(user_query)


# === STEP 4: Streamlit UI ===

st.title("🧾 Time Card Agent")

uploaded_file = st.file_uploader("Upload Time Card Excel", type=["xlsx"])
uploaded_file_pdf = st.file_uploader("Upload Time Card pdf", type=["pdf"])


if uploaded_file:
    timecard = extract_timecard_from_excel(uploaded_file, "Apr")
    st.text_area("sheet names", timecard, height=200)
    timecard_data = map_extracted_text_to_timecard_data_with_confidence_score(timecard) if timecard else None
    st.subheader("📋 Extracted Time Card Data")
    st.code(timecard_data, language="json")
    st.json(timecard_data)

if uploaded_file_pdf:
    timecard_text = robust_extract_text(uploaded_file_pdf)
    st.subheader("📋 Extracted Time Card Data")
    st.text_area("Extracted Text from PDF", timecard_text, height=200)
    timecard_data = map_extracted_text_to_timecard_data_with_confidence_score(timecard_text) if timecard_text else None
    st.subheader("📌 Extracted Fields")
    st.code(timecard_data, language="json")
    st.json(timecard_data)

    query = st.text_input("💬 Ask a question about the time card", value="Was the employee compliant with work hour rules?")
    if query and st.button("Run Agent"):
        response = run_time_card_agent(query, parsed_data)
        st.markdown("### 🤖 Agent Response")
        st.write(response)
