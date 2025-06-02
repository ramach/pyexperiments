from langchain.tools import tool
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

# Create the SQLDatabase instance
engine = create_engine("sqlite:///ap_ar.db")
db = SQLDatabase(engine)

@tool("query_ap_ar_db", return_direct=True)
def query_ap_ar_db(question: str) -> str:
    """Answer AP/AR invoice-related questions by querying the database."""
    try:
        result = db.run(question)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
