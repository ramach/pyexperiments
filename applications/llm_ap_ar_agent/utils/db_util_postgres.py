import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import os

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "invoice_ai"),
    "user": os.getenv("DB_USER", "testuser"),
    "password": os.getenv("DB_PASS", "qwerty"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

def get_connection():
    return psycopg2.connect(**DB_PARAMS)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id SERIAL PRIMARY KEY,
            table_name TEXT NOT NULL,
            vendor_name TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            correction_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def insert_correction(table_name, vendor_name, customer_name,  correction_text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO corrections (table_name, vendor_name, customer_name, correction_text)
        VALUES (%s, %s, %s, %s, %s)
    """, (table_name, vendor_name, customer_name, correction_text))
    conn.commit()
    conn.close()

def get_last_correction(vendor_name, field_name):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT new_value FROM corrections
        WHERE vendor_name = %s AND field_name = %s
        ORDER BY timestamp DESC LIMIT 1;
    """, (vendor_name, field_name))
    row = cur.fetchone()
    conn.close()
    return row["new_value"] if row else None
