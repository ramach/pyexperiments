# tools/time_card_parser_tool.py

from langchain.tools import Tool
from typing import Callable

import pandas as pd
import json
from datetime import datetime

def parse_time_card(filepath: str) -> dict:
    # Load the Excel file
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    sheet = xl.parse(xl.sheet_names[0], header=None)

    result = {}

    # Extract employee and manager details (assumed in first few rows)
    for i in range(10):  # Adjust range based on actual structure
        row = sheet.iloc[i].dropna().astype(str).tolist()
        if len(row) >= 2:
            key, value = row[0].lower(), row[1]
            if "employee" in key:
                result["employee_name"] = value
            elif "manager" in key:
                result["manager_name"] = value
            elif "contact" in key or "phone" in key:
                result["contact"] = value
            elif "hourly rate" in key:
                result["hourly_rate"] = value
            elif "signature" in key:
                result["signature"] = value

    # Search for hours worked and compute total
    total_hours = 0.0
    for i in range(len(sheet)):
        row = sheet.iloc[i]
        for val in row:
            try:
                # Allow floats or ints that might be hours
                if isinstance(val, (int, float)) and 0 < val < 24:
                    total_hours += float(val)
            except Exception:
                pass

    result["total_hours"] = round(total_hours, 2)

    # Calculate total amount (if hourly_rate present)
    try:
        rate = float(result.get("hourly_rate", "0").replace("$", ""))
        result["total_amount"] = round(total_hours * rate, 2)
    except Exception:
        result["total_amount"] = "N/A"

    return result

if __name__ == "__main__":
    filepath = "/Users/krishnaramachandran/Downloads/Timecards/2-16-2016.xlsx"  # Replace with your actual file path
    data = parse_time_card(filepath)

    # Convert datetime to string if any
    def json_safe(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    print(json.dumps(data, indent=2, default=json_safe))

def get_time_card_parser_tool(filepath: str) -> Tool:
    def tool_func(_input: str) -> str:
        data = parse_time_card(filepath)
        return json.dumps(data, indent=2)

    return Tool(
        name="TimeCardParser",
        func=tool_func,
        description=(
            "Use this tool to extract structured details like employee name, manager name, total hours, "
            "hourly rate, and total amount from a time card Excel file. Input can be a user query like "
            "'Get timecard details'."
        )
    )
