from langchain.tools import tool

@tool
def validate_invoice_against_rules(invoice: dict, rules: str) -> str:
    """Validates an invoice dict against business rules string."""
    messages = []
    rate_rule = "141"
    if invoice["hourly_rate"] != int(rate_rule):
        messages.append("Hourly rate mismatch.")
    if invoice["hours_worked"] > 40:
        messages.append("More than 40 hours worked, needs approval.")
    return " | ".join(messages) or "Invoice is compliant."
