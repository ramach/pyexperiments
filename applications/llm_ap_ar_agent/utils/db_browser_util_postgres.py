import psycopg
from psycopg.rows import dict_row
import os

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "invoice_ai"),
    "user": os.getenv("DB_USER", "testuser"),
    "password": os.getenv("DB_PASS", "qwerty"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

def get_connection():
    return psycopg.connect(**DB_PARAMS, row_factory=dict_row)

def list_all_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT schemaname, tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename;
            """)
            return [f"{row['schemaname']}.{row['tablename']}" for row in cur.fetchall()]

def preview_table(schema_and_table: str, limit: int = 100):
    schema, table = schema_and_table.split(".")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM "{schema}"."{table}" LIMIT {limit};')
            return cur.fetchall()

def run_custom_query(query: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
