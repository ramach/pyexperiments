from langchain.tools import tool
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

# Connect to the SQLite database
engine = create_engine("sqlite:///ap_ar.db")
db = SQLDatabase(engine)

@tool("query_ap_ar_db", return_direct=True)
def query_ap_ar_db(question: str) -> str:
    """
    Use SQL to answer AP/AR-related questions involving invoices, vendors, and payments.
    Supports multi-table joins and aggregation.
    """
    try:
        result = db.run(question)
        return result
    except Exception as e:
        return f"Database query error: {e}"
