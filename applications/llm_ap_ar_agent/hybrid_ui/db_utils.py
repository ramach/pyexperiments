import sqlite3
from datetime import datetime
import pandas as pd
import json

DB_PATH = "invoice_data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT,
                invoice_date TEXT,
                due_date TEXT,
                vendor TEXT,
                client TEXT,
                total_amount TEXT,
                file_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS toolchain_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT,
                verification TEXT,
                po_match TEXT,
                approval TEXT,
                payment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gpt_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                guidance TEXT,
                output_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT,
                field_name TEXT,
                original_value TEXT,
                corrected_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_invoice_and_results(invoice_data, toolchain_data):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO invoices (invoice_id, invoice_date, due_date, vendor, client, total_amount, file_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            invoice_data.get("invoice_id"),
            invoice_data.get("invoice_date"),
            invoice_data.get("due_date"),
            invoice_data.get("vendor"),
            invoice_data.get("client"),
            invoice_data.get("total_amount"),
            invoice_data.get("file_name", "")
        ))
        conn.commit()

        cur.execute("""
            INSERT INTO toolchain_results (invoice_id, verification, po_match, approval, payment)
            VALUES (?, ?, ?, ?, ?)
        """, (
            invoice_data.get("invoice_id"),
            str(toolchain_data.get("verification")),
            str(toolchain_data.get("po_match")),
            str(toolchain_data.get("approval")),
            str(toolchain_data.get("payment"))
        ))
        conn.commit()

def log_gpt_mapping(input_text, guidance, output_dict):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO gpt_logs (input_text, guidance, output_json)
            VALUES (?, ?, ?)
        """, (input_text, guidance, json.dumps(output_dict)))
        conn.commit()

def save_corrections(invoice_id, original: dict, corrected: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for field in corrected:
            o = original.get(field, "")
            c = corrected.get(field, "")
            if o != c:
                cur.execute("""
                    INSERT INTO corrections (invoice_id, field_name, original_value, corrected_value)
                    VALUES (?, ?, ?, ?)
                """, (invoice_id, field, o, c))
        conn.commit()

def fetch_all_invoices():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)

def fetch_corrections_for_field(field_name):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT invoice_id, original_value, corrected_value, created_at FROM corrections WHERE field_name = ? ORDER BY created_at DESC",
            conn, params=(field_name,)
        )