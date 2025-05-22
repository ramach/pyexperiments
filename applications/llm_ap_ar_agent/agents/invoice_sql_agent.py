import os
import sqlite3
from sqlalchemy import create_engine, text
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase

# Set your OpenAI key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# === Step 1: Create SQLite DB and sample table ===
db_file = "ap_ar.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Drop if exists and create
cursor.execute("DROP TABLE IF EXISTS invoices")
cursor.execute("""
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor TEXT,
    invoice_number TEXT,
    amount REAL,
    due_date TEXT,
    status TEXT
)
""")

# Insert mock data
sample_data = [
    ("Vendor ABC", "INV-1001", 1200.50, "2024-05-01", "unpaid"),
    ("Vendor XYZ", "INV-1002", 800.00, "2024-04-15", "paid"),
    ("Vendor ABC", "INV-1003", 400.00, "2024-06-01", "unpaid"),
    ("Vendor DEF", "INV-1004", 950.00, "2024-05-15", "unpaid"),
]
cursor.executemany("INSERT INTO invoices (vendor, invoice_number, amount, due_date, status) VALUES (?, ?, ?, ?, ?)", sample_data)
conn.commit()
conn.close()

# === Step 2: Load DB with LangChain SQLDatabase ===
engine = create_engine(f"sqlite:///{db_file}")
db = SQLDatabase(engine)

# === Step 3: Create SQL Agent ===
llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-3.5-turbo"
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# === Step 4: Ask questions in natural language ===
query = "What is the total amount of unpaid invoices?"
response = agent_executor.run(query)
print("\n📄 Answer:", response)
