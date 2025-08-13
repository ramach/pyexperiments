import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from utils.db_browser_util_postgres import list_all_tables, preview_table, run_custom_query

tab1, tab2 = st.tabs(["📄 SOW Mapping", "🧭 Admin: DB Explorer"])

with tab2:
    st.header("🧭 Database Table Explorer")

    # Dropdown to list all tables
    all_tables = list_all_tables()
    selected_table = st.selectbox("🔽 Select a table to preview", options=all_tables)

    if selected_table:
        st.subheader(f"Preview of `{selected_table}`")
        rows = preview_table(selected_table)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Table is empty.")

    st.divider()

    st.subheader("🧠 Run Custom SQL Query")
    sql = st.text_area("Write a SQL query (SELECT only recommended)", height=120)

    if st.button("▶️ Run Query"):
        try:
            result = run_custom_query(sql)
            df = pd.DataFrame(result)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")
