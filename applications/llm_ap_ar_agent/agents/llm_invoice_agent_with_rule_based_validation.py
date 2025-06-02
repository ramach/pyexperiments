from datetime import datetime
from typing import Dict, Any
from langchain.tools import tool


def validate_invoice(invoice: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates an invoice against dynamically supplied rules.
    """
    results = {}

    for rule_name, rule_def in rules.items():
        check_fn = rule_def.get("check")
        reason_fn = rule_def.get("reason")

        try:
            passed = check_fn(invoice)
            reason = reason_fn(invoice) if not passed else "OK"
        except Exception as e:
            passed = False
            reason = f"Error in rule logic: {str(e)}"

        results[rule_name] = {
            "Pass": passed,
            "Reason": reason
        }

    return results


# Default rules
default_rules = {
    "Rule 0 - Payment Terms": {
        "check": lambda inv: inv["hourly_rate"] <= 141 and
                             inv["total_hours"] <= 1976 and
                             (inv["weekly_hours"] <= 40 or inv.get("prior_approval", False)) and
                             inv.get("timesheet_approved", False),
        "reason": lambda inv: "; ".join([
            "Hourly rate exceeds $141/hr." if inv["hourly_rate"] > 141 else "",
            "Total hours exceed 1,976." if inv["total_hours"] > 1976 else "",
            "More than 40 hours/week without prior approval." if inv["weekly_hours"] > 40 and not inv.get("prior_approval", False) else "",
            "Timesheet is not client-approved." if not inv.get("timesheet_approved", False) else ""
        ]).strip("; ")
    },
    "Rule 1 - Expenses": {
        "check": lambda inv: inv["expenses"] == 0 or inv.get("receipts_attached", False),
        "reason": lambda inv: "Receipts missing for submitted expenses." if inv["expenses"] > 0 and not inv.get("receipts_attached", False) else "OK"
    },
    "Rule 2 - Invoicing & Payment Terms": {
        "check": lambda inv: abs((datetime.strptime(inv["invoice_date"], "%Y-%m-%d") -
                                  datetime.strptime(inv["last_friday_of_month"], "%Y-%m-%d")).days) <= 5 and
                             inv.get("payment_terms_days", 30) <= 30,
        "reason": lambda inv: "; ".join([
            "Invoice date is not close to last Friday of month."
            if abs((datetime.strptime(inv["invoice_date"], "%Y-%m-%d") -
                    datetime.strptime(inv["last_friday_of_month"], "%Y-%m-%d")).days) > 5 else "",
            "Payment terms exceed 30 days." if inv.get("payment_terms_days", 30) > 30 else ""
        ]).strip("; ")
    }
}


@tool
def llm_invoice_agent(invoice: Dict[str, Any], rules: Dict[str, Any] = default_rules) -> Dict[str, Any]:
    """
    Validates a contractor invoice against dynamically supplied rules.
    """
    return validate_invoice(invoice, rules)
