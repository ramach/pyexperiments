import json
import random
from datetime import datetime, timedelta

def generate_synthetic_invoice(n=100):
    vendors = ["Acme Corp", "Globex Inc", "Initech", "Umbrella LLC"]
    records = []
    for i in range(n):
        inv_id = f"INV-{random.randint(1000,9999)}"
        vendor = random.choice(vendors)
        amount = round(random.uniform(500, 20000), 2)
        date = (datetime.today() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        po = f"PO-{random.randint(1000,9999)}"
        text = f"""
        This invoice is issued by {vendor} on {date}.
        Invoice ID: {inv_id}
        The amount due is ${amount}. This invoice references purchase order {po}.
        """
        response = {
            "invoice_id": inv_id,
            "vendor": vendor,
            "amount": amount,
            "date": date,
            "po_number": po
        }
        records.append({
            "prompt": f"Extract invoice_id, vendor, amount, date, po_number from:\n{text}",
            "response": json.dumps(response)
        })
    return records

# Save to file
synthetic_data = generate_synthetic_invoice(100)
with open("flan_invoice_train.jsonl", "w") as f:
    for r in synthetic_data:
        f.write(json.dumps(r) + "\n")

print("✅ Generated flan_invoice_train.jsonl")
