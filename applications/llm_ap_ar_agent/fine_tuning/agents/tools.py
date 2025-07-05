def verify_invoice(invoice: dict) -> str:
    if not invoice.get("invoice_id"):
        return "Missing invoice_id"
    if invoice.get("total") <= 0:
        return "Invalid total"
    return "Invoice verification passed"

def check_business_rules(invoice: dict, rules: str) -> str:
    violations = []
    if "hours" in invoice and invoice["hours"] > 40:
        if "prior approval" not in rules.lower():
            violations.append("Worked over 40 hours without approval.")
    return "Compliant" if not violations else f"Non-compliant: {', '.join(violations)}"
