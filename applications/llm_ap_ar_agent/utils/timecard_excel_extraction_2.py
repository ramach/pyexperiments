import pandas as pd
import re

def clean_label(label_text):
    """Cleans the label text to be used as a dictionary key."""
    if not isinstance(label_text, str):
        return None
    return label_text.replace(':', '').replace('[', '').replace(']', '').strip().lower().replace(' ', '_')

def extract_value_from_row(row_series):
    """Finds the most likely value in a row, preferring the last numeric value."""
    numeric_value = None
    last_value = None

    for value in reversed(row_series.tolist()):
        if pd.notna(value):
            try:
                if isinstance(value, str):
                    cleaned_val = re.sub(r'[$\s,]', '', value)
                    if cleaned_val:
                        numeric_value = float(cleaned_val)
                else:
                    numeric_value = float(value)
                break
            except (ValueError, TypeError):
                if last_value is None:
                    last_value = value

    return numeric_value if numeric_value is not None else last_value

def parse_timecard_dynamically(excel_file_path, sheet_name=0):
    """
    Parses a timecard from a given Excel file by dynamically finding labels and values.

    This function reads your specified Excel file, intelligently scans it for data fields,
    and returns "MISSING" for any required field it cannot find.

    Args:
        excel_file_path (str): The full path to YOUR Excel timecard file.
        sheet_name (str or int, optional): The name or index of the sheet. Defaults to 0.

    Returns:
        dict: A dictionary of the extracted fields. Values for unfound fields are set to "MISSING".
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)

        # --- Phase 1: Scan the sheet to find all possible key-value data ---
        extracted_data = {}

        # Scan for key:value pairs (e.g., "Manager: Ian Gomez")
        for r_idx, row in df.iterrows():
            for c_idx, cell_value in enumerate(row):
                if isinstance(cell_value, str) and ':' in cell_value:
                    label = clean_label(cell_value)
                    if c_idx + 1 < len(row) and pd.notna(row.iloc[c_idx + 1]):
                        value = row.iloc[c_idx + 1]
                        if label and label not in extracted_data:
                            extracted_data[label] = value

        # Scan for summary table rows (e.g., "Total hours", "Rate per hour")
        summary_labels = {
            "total_hours": "Total hours",
            "rate_per_hour": "Rate per hour",
            "total_pay": "Total pay"
        }
        for label_key, label_text in summary_labels.items():
            for r_idx, row in df.iterrows():
                row_text = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
                if label_text in row_text:
                    value = extract_value_from_row(row)
                    extracted_data[label_key] = value
                    break

        # Scan for special cases like "Employee Name" which might not have a colon
        for r_idx, row in df.iterrows():
            if "Employee" in str(row.iloc[0]) and "name" not in str(row.iloc[0]).lower() and pd.notna(row.iloc[1]):
                if 'employee_name' not in extracted_data:
                    extracted_data['employee_name'] = row.iloc[1]
                    break

        # --- Phase 2: Consolidate and build the final, clean output dictionary ---
        found_values = {}

        # Consolidate various possible keys into standard keys
        found_values['employee_name'] = extracted_data.get('employee') or extracted_data.get('employee_name')
        found_values['manager_name'] = extracted_data.get('manager')
        found_values['total_hours'] = extracted_data.get('total_hours')
        found_values['rate_per_hour'] = extracted_data.get('rate_per_hour')
        found_values['amount'] = extracted_data.get('total_pay')

        # Combine address fields
        street = extracted_data.get('street_address')
        city_info = extracted_data.get('code')
        if street and city_info:
            found_values['address'] = f"{street}, {city_info}"
        elif street:
            found_values['address'] = street
        else:
            found_values['address'] = None

        # Define the final structure and substitute "MISSING" for any blank or unfound values
        final_data = {}
        required_fields = ['employee_name', 'address', 'manager_name', 'total_hours', 'rate_per_hour', 'amount']

        for field in required_fields:
            value = found_values.get(field)
            if value is None or pd.isna(value):
                final_data[field] = "MISSING"
            else:
                final_data[field] = value

        return final_data

    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"The file was not found at the path: '{excel_file_path}'")
        print("Please make sure the file path is correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- HOW TO USE THIS SCRIPT ---
if __name__ == "__main__":

    # --- STEP 1: Create a sample Excel file for demonstration ---
    # This part is ONLY for creating an example file.
    # In your real code, you can remove this section.
    # We will intentionally leave out the 'Manager' field to test the "MISSING" functionality.
    print("--- Creating a sample Excel file for demonstration ---")
    sample_file_path = '/Users/krishnaramachandran/pyexperiments/applications/llm_ap_ar_agent/mockdata/samples/suhas/timecard.xlsx'
    sample_file_path_output = '/Users/krishnaramachandran/pyexperiments/applications/llm_ap_ar_agent/mockdata/samples/suhas/timecard2.xlsx'
    list_of_rows = [
        ["Employee", "Lovina Lalye"],
        ["[Street Address]:", "422 Timor Terr"],
        ["Code]:", "sunnyvale, CA, 94089"],
        # Notice the "Manager:" line is commented out to test 'MISSING'
        # ["Manager:", "Ian Gomez"],
        ["Week ending:", "12/4/2016"],
        [],
        ["Day", "Date", "Regular Hours", "Total"],
        ["Monday", "11/28/2016", 151.00, 151.00], # Using 151 hours as per your example
        ["Total hours", None, "151.00", "151.00"],
        ["Rate per hour", None, "$60"],
        ["Total pay", None, "$ 9,060.00"]
    ]
    dummy_df = pd.DataFrame(list_of_rows)
    dummy_df.to_excel(sample_file_path_output, sheet_name="TimecardData", index=False, header=False)
    print(f"Sample file created at: '{sample_file_path}'")
    print("This file is missing the 'Manager' field to demonstrate the 'MISSING' feature.\n")

    # --- STEP 2: Run the parser on your Excel file ---
    # IMPORTANT: Replace `sample_file_path` with the path to your real timecard file.
    print(f"--- Parsing data from '{sample_file_path}' ---")

    # You can specify a sheet name if needed, for example: sheet_name="MyTimecardSheet"
    timecard_data = parse_timecard_dynamically(sample_file_path, sheet_name="Mar")

    # --- STEP 3: Display the results ---
    if timecard_data:
        print("\n--- Dynamically Extracted Timecard Metadata ---")
        for key, value in timecard_data.items():
            display_key = key.replace('_', ' ').title()
            print(f"  {display_key}: {value}")
        print("---------------------------------------------")
    else:
        print("\nCould not parse the timecard due to an error.")