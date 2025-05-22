import json

import pandas as pd
from typing import List, Dict, Any
import openpyxl
from pandas.core.interchange.dataframe_protocol import DataFrame


def parse_time_card_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses time card data from an Excel file.

    Assumes the Excel sheet has columns like:
    - Employee Name
    - Date
    - Project
    - Hours Worked
    - Task Description (optional)

    Args:
        file_path (str): Path to the Excel (.xlsx) file.

    Returns:
        List[Dict[str, Any]]: A list of time card entries.
    """
    try:
        df = pd.read_excel(file_path)
        print(df.head(40))  # Show first 20 rows
        records = df.to_dict(orient='records')
        return records
    except Exception as e:
        raise ValueError(f"Failed to parse time card Excel file: {e}")

from datetime import date, datetime

def convert_dates_to_str(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()  # or str(obj)
    if isinstance(obj, list):
        return [convert_dates_to_str(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_dates_to_str(value) for key, value in obj.items()}
    return obj

import pandas as pd
from typing import Dict, Any
import re

import pandas as pd
from typing import Dict, Any
import re

def extract_timecard_metadata_generic(file_path: str) -> Dict[str, Any]:
    metadata = {}
    df = pd.read_excel(file_path)
    # Known field patterns (regex-style)
    field_patterns = {
        "employee_name": r"\bemployee\b",
        "manager_name": r"\bmanager\b",
        "street_address": r"street address",
        "address_2": r"address 2",
        "city_state_zip": r"city.*zip",
        "phone": r"phone",
        "email": r"e-?mail",
        "week_ending": r"week ending",
        "total_hours": r"total hours?",
        "hourly_rate": r"rate per hour",
        "total_amount": r"total pay|total amount",
    }

    def match_field(label: str) -> str:
        label_lower = label.strip().lower()
        for field, pattern in field_patterns.items():
            if re.search(pattern, label_lower):
                return field
        return None

    # Scan row-by-row
    for _, row in df.iterrows():
        row = row.fillna("").astype(str).str.strip().tolist()
        for i in range(len(row) - 1):
            field_name = match_field(row[i])
            if field_name and row[i + 1]:
                value = row[i + 1]
                # Try converting numerical fields
                if field_name in ["total_hours", "hourly_rate", "total_amount"]:
                    try:
                        value = float(re.sub(r"[^\d.]", "", value))
                    except:
                        value = 0.0
                metadata[field_name] = value

    # Build full address
    full_address = ", ".join(filter(None, [
        metadata.get("street_address", ""),
        metadata.get("address_2", ""),
        metadata.get("city_state_zip", "")
    ]))

    return {
        "employee_name": metadata.get("employee_name", "Unknown"),
        "manager_name": metadata.get("manager_name", "Unknown"),
        "phone": metadata.get("phone", "Unknown"),
        "email": metadata.get("email", "Unknown"),
        "address": full_address or "Unknown",
        "week_ending": metadata.get("week_ending", "Unknown"),
        "total_hours": metadata.get("total_hours", 0.0),
        "hourly_rate": metadata.get("hourly_rate", 0.0),
        "total_amount": metadata.get("total_amount", 0.0),
    }

from typing import Dict
import pandas as pd
import re
from langchain.tools import tool

#@tool
from typing import Dict
import pandas as pd
import re
from langchain.tools import tool

#@tool
def extract_employee_timecard_info(file_path: str) -> Dict:
    data = pd.read_excel(file_path)
    """
    Extracts employee and manager timecard metadata from a structured Excel sheet.
    Handles employee name, address, email, phone, manager, hours, rate, and total pay.
    """

    result = {
        "employee_name": None,
        "manager_name": None,
        "phone": None,
        "email": None,
        "address": None,
        "week_ending": None,
        "total_hours": 0.0,
        "hourly_rate": 0.0,
        "total_amount": 0.0,
    }

    def extract_cell_value(row, key_substring):
        for idx, cell in enumerate(row):
            if isinstance(cell, str) and key_substring.lower() in cell.lower():
                if idx + 1 < len(row):
                    return str(row[idx + 1]).strip()
        return None

    for i, row in data.iterrows():
        row_list = row.tolist()

        # Look for key fields in this row
        row_str = " ".join(str(cell) for cell in row_list if pd.notna(cell)).strip()

        if "employee" in row_str.lower() and not result["employee_name"]:
            result["employee_name"] = extract_cell_value(row_list, "employee")

        if "manager" in row_str.lower() and not result["manager_name"]:
            result["manager_name"] = extract_cell_value(row_list, "manager")

        if "phone" in row_str.lower() and not result["phone"]:
            phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", row_str)
            if phone_match:
                result["phone"] = phone_match.group()

        if "email" in row_str.lower() and not result["email"]:
            email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", row_str)
            if email_match:
                result["email"] = email_match.group()

        if "street address" in row_str.lower():
            result["address"] = extract_cell_value(row_list, "street address")
        if "city" in row_str.lower() and result["address"]:
            city = extract_cell_value(row_list, "city")
            if city:
                result["address"] += ", " + city

        if "week ending" in row_str.lower() and not result["week_ending"]:
            date_match = re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", row_str)
            if date_match:
                result["week_ending"] = date_match.group()

        if "total hours" in row_str.lower():
            float_matches = re.findall(r"\b\d+\.\d+\b", row_str)
            if float_matches:
                result["total_hours"] = float(float_matches[0])

        if "rate per hour" in row_str.lower():
            match = re.search(r"\$?\s*([\d,.]+)", row_str)
            if match:
                result["hourly_rate"] = float(match.group(1).replace(",", ""))

        if "total pay" in row_str.lower():
            match = re.search(r"\$?\s*([\d,]+\.\d+)", row_str)
            if match:
                result["total_amount"] = float(match.group(1).replace(",", ""))

    return result


import pandas as pd
import re

def extract_summary_fields_from_timecard(file_path: str) -> dict:
    df = pd.read_excel(file_path)
    result = {
        "total_hours": None,
        "hourly_rate": None,
        "total_amount": None,
        "employee_name": None
    }

    def parse_amount(value: str) -> float:
        return float(re.sub(r"[^\d.]", "", value)) if value else 0.0

    for _, row in df.iterrows():
        joined = " ".join(str(cell) for cell in row if pd.notna(cell)).lower()

        if "total hours" in joined:
            matches = re.findall(r"\b\d+\.\d+\b", joined)
            if matches:
                result["total_hours"] = float(matches[0])

        elif "rate per hour" in joined:
            match = re.search(r"\$?\s?([\d,.]+)", joined)
            if match:
                result["hourly_rate"] = parse_amount(match.group(1))

        elif "total pay" in joined:
            match = re.search(r"\$?\s?([\d,]+\.\d+)", joined)
            if match:
                result["total_amount"] = parse_amount(match.group(1))

        elif "signature" in joined and not result["employee_name"]:
            # Heuristic: pick non-empty text in the same row before 'employee signature'
            possible_names = [cell for cell in row if isinstance(cell, str) and cell.strip()]
            if possible_names:
                result["employee_name"] = possible_names[0]

    return result


if __name__ == "__main__":
    parsed_time_card = \
        parse_time_card_excel("/Users/krishnaramachandran/kasu.ai/pyexperiments/llm_ap_ar_agent/sample-timesheet.xlsx")
    # Convert all datetime/date objects to strings
    clean_data = convert_dates_to_str(parsed_time_card)
    json_output = json.dumps(clean_data, indent=4)
    print(json_output)

