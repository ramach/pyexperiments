import json
import random

vendors = ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Ltd", "Epsilon Co"]
statuses = ["valid", "exceeds max allowed amount", "requires approval", "is compliant with policy", "is non-compliant"]
amounts = [500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]

data = []

for i in range(100):
    inv_id = f"INV{1000 + i}"
    vendor = random.choice(vendors)
    amount = random.choice(amounts)

    if amount > 10000:
        response = f"Invoice {inv_id} exceeds max allowed amount."
    else:
        response = f"Invoice {inv_id} is valid."

    data.append({
        "prompt": f"Validate invoice {inv_id} from {vendor} with amount ${amount}",
        "response": response
    })

# Save to file
with open("sample_ap_ar_100.jsonl", "w") as f:
    for record in data:
        f.write(json.dumps(record) + "\n")

print("✅ Generated sample_ap_ar_100.jsonl")
