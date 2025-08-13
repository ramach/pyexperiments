import psycopg
from psycopg.rows import dict_row
import os

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "invoice_ai"),
    "user": os.getenv("DB_USER", "testuser"),
    "password": os.getenv("DB_PASS", "qwerty"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": 5432,
}

def get_connection():
    return psycopg.connect(**DB_PARAMS, row_factory=dict_row)

def init_corrections_table():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS guidance (
                    id SERIAL PRIMARY KEY,
                    contracting_company TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    correction TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

def insert_sow_correction(company: str, vendor: str, table_name:str, correction: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO guidance (contracting_company, vendor, table_name, correction)
                VALUES (%s, %s, %s, %s);
            """, (company, vendor, table_name, correction))

def get_last_sow_correction(company: str, vendor: str, table_name:str) -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT correction FROM guidance
                WHERE contracting_company = %s AND vendor = %s AND table_name = %s
                ORDER BY timestamp DESC LIMIT 1;
            """, (company, vendor, table_name))
            row = cur.fetchone()
            return row["correction"] if row else ""

