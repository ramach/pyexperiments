# db_utils.py (psycopg3 version)

import os
import psycopg
from psycopg.rows import dict_row
from datetime import datetime

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "invoice_ai"),
    "user": os.getenv("DB_USER", "testuser"),
    "password": os.getenv("DB_PASS", "qwerty"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

def get_connection():
    return psycopg.connect(**DB_PARAMS, row_factory=dict_row)

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id SERIAL PRIMARY KEY,
                    vendor_name TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    correction_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

def insert_correction(vendor_name, field_name, old_value, new_value, correction_text):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO corrections (vendor_name, field_name, old_value, new_value, correction_text)
                VALUES (%s, %s, %s, %s, %s);
            """, (vendor_name, field_name, old_value, new_value, correction_text))

def get_last_correction(vendor_name, field_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT new_value FROM corrections
                WHERE vendor_name = %s AND field_name = %s
                ORDER BY timestamp DESC LIMIT 1;
            """, (vendor_name, field_name))
            row = cur.fetchone()
            return row["new_value"] if row else None

def get_corrections(vendor_filter: str = None) -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if vendor_filter:
                cur.execute("""
                    SELECT vendor_name, field_name, old_value, new_value, correction_text, timestamp
                    FROM corrections
                    WHERE vendor_name ILIKE %s
                    ORDER BY timestamp DESC;
                """, (f"%{vendor_filter}%",))
            else:
                cur.execute("""
                    SELECT vendor_name, field_name, old_value, new_value, correction_text, timestamp
                    FROM corrections
                    ORDER BY timestamp DESC;
                """)
            return cur.fetchall()


