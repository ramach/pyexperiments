# chains/scaffold_ap_chain.py
import json
import os
from datetime import datetime


def load_ap_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "synthetic", "ap_ar_data.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data.get("accounts_payable", [])


def run_ap_chain(query: str) -> str:
    data = load_ap_data()
    query = query.lower()
    results = []

    for item in data:
        vendor = item["vendor"].lower()
        status = item["status"].lower()
        due_date = datetime.strptime(item["due_date"], "%Y-%m-%d")
        today = datetime.today()

        if "overdue" in query and status == "unpaid" and due_date < today:
            results.append(item)
        elif "unpaid" in query and status == "unpaid":
            results.append(item)
        elif vendor in query:
            results.append(item)

    if not results:
        return "No AP records matched the query."

    return "\n".join([
        f"Vendor: {r['vendor']}, Invoice: {r['invoice_id']}, Due: {r['due_date']}, Status: {r['status']}, Amount: ${r['amount']:.2f}"
        for r in results
    ])