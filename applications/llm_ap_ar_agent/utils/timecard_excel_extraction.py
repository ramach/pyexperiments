import pandas as pd
import re # For cleaning currency symbols

def parse_timecard_from_excel(excel_file_path, sheet_name=0):
    """
    Parses a timecard from an Excel file (with layout similar to the provided PDF)
    to extract metadata.

    Args:
        excel_file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The name or index of the sheet
                                           Defaults to 0 (the first sheet).

    Returns:
        dict: A dictionary containing the extracted metadata,
              or None if an error occurs.
    """
    try:
        # Read the Excel file. header=None is important if the actual data
        # doesn't start with a clear header row at the very top, or if labels
        # are scattered.
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)

        metadata = {}

        # Helper function to find a value next to a label
        def find_value_by_label(dataframe, label_text):
            for r_idx, row in dataframe.iterrows():
                for c_idx, cell_value in enumerate(row):
                    if isinstance(cell_value, str) and label_text in cell_value:
                        # Try to get value from the next cell in the same row
                        if c_idx + 1 < len(row):
                            return row.iloc[c_idx + 1]
            return None

        # Helper function to clean currency and convert to float
        def clean_currency(value_str):
            if isinstance(value_str, (int, float)):
                return float(value_str)
            if isinstance(value_str, str):
                # Remove currency symbols, commas, and whitespace
                cleaned_value = re.sub(r'[$\s,]', '', value_str)
                try:
                    return float(cleaned_value)
                except ValueError:
                    return None
            return None

        # --- Metadata Extraction based on PDF sample ---
        # These are assumptions. Actual cell locations might vary.
        # The .iloc calls are examples if you know exact positions.
        # The find_value_by_label is more flexible if labels are consistent.

        # Employee Name: "Lovina Lalye" was next to "Employee"
        metadata['employee_name'] = find_value_by_label(df, "Employee")
        # If fixed: e.g., df.iloc[0, 1] (assuming "Employee" in A1, Name in B1)

        # Employee Address: Combine street and city/zip
        # "422 Timor Terr" was next to "[Street Address]"
        # "sunnyvale, CA, 94089" was next to "Code]"
        street_address = find_value_by_label(df, "[Street Address]")
        city_state_zip = find_value_by_label(df, "Code]")
        if street_address and city_state_zip:
            metadata['employee_address'] = f"{street_address}, {city_state_zip}"
        elif street_address:
            metadata['employee_address'] = street_address
        else:
            metadata['employee_address'] = None
        # If fixed: e.g., address_part1 = df.iloc[1,1], address_part2 = df.iloc[3,1]

        # Manager Name: "Ian Gomez" was next to "Manager:"
        metadata['manager_name'] = find_value_by_label(df, "Manager:")
        # If fixed: e.g., df.iloc[row_idx_manager, col_idx_manager_value]

        # --- Locating the main data table for hours, rate, pay ---
        # This part is trickier as tables can vary.
        # We'll search for rows containing key labels from the PDF's table section.

        total_hours_val = None
        rate_per_hour_val = None
        total_pay_val = None

        for r_idx, row in df.iterrows():
            row_str = ' '.join(map(str, row.dropna().tolist())).lower() # Join cells for easier search

            if "total hours" in row_str:
                # Assuming the value is in one of the last columns of that row
                # In the PDF, "30.00" appeared in the 3rd and last column of "Total hours" row
                # Try to get the last numeric value in that row
                for cell_val in reversed(row.tolist()):
                    numeric_val = clean_currency(cell_val) # Use clean_currency to handle strings
                    if numeric_val is not None:
                        total_hours_val = numeric_val
                        break

            if "rate per hour" in row_str:
                # In PDF, "$60" was in the 3rd column of its row
                for cell_val in row.tolist():
                    numeric_val = clean_currency(cell_val)
                    if numeric_val is not None:
                        rate_per_hour_val = numeric_val
                        break # Take the first numeric value found after label

            if "total pay" in row_str:
                # In PDF, "$ 1,800.00" was in 3rd and last column
                for cell_val in reversed(row.tolist()):
                    numeric_val = clean_currency(cell_val)
                    if numeric_val is not None:
                        total_pay_val = numeric_val
                        break

        metadata['total_hours'] = total_hours_val
        metadata['rate_per_hour'] = rate_per_hour_val

        # If total pay is directly found, use it. Otherwise, try to calculate.
        if total_pay_val is not None:
            metadata['amount'] = total_pay_val
        elif total_hours_val is not None and rate_per_hour_val is not None:
            metadata['amount'] = total_hours_val * rate_per_hour_val
        else:
            metadata['amount'] = None

        return metadata

    except FileNotFoundError:
        print(f"Error: The file '{excel_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- How to use the parser ---
if __name__ == "__main__":
    # Replace 'your_timecard.xlsx' with the actual path to your Excel file
    # that has a layout SIMILAR to the provided PDF.
    file_path = '/Users/krishnaramachandran/pyexperiments/applications/llm_ap_ar_agent/mockdata/samples/lisa/timecard.xlsx'
    sheet = 0 # Or your specific sheet name/index

    # For testing, you would create 'your_timecard_layout_sample.xlsx'
    # with data arranged like the PDF. For instance:
    # A1: Employee, B1: Lovina Lalye
    # A2: [Street Address], B2: 422 Timor Terr
    # A3: [Address 2], B3: (empty or an apt number)
    # A4: Code], B4: sunnyvale, CA, 94089
    # A5: Week ending:, B5: 12/4/2016
    # A7: Manager:, B7: Ian Gomez
    # ... (other fields like phone, email)
    # Then the table, perhaps starting around row 10:
    # A10: Day, B10: Date, C10: Regular Hours, ..., G10: Total
    # ... rows for Monday-Sunday ...
    # A17: Total hours, C17: 30.00, G17: 30.00
    # A18: Rate per hour, C18: $60
    # A19: Total pay, C19: $ 1,800.00, G19: $1,800.00

    # Create a dummy Excel file for testing based on the PDF structure
    try:
        # This is a simplified representation for the dummy file
        data_for_excel = {
            0: ["Employee", "[Street Address]", "[Address 2]", "Code]", "Week ending:", None, "Manager:", "phone:", "mail:"],
            1: ["Lovina Lalye", "422 Timor Terr", None, "sunnyvale, CA, 94089", "12/4/2016", None, "Ian Gomez", "650-793-1285", "lovina@datatorrent.com"],
            2: [None]*9, # Empty row for spacing
            3: ["Day", "Date", "Regular Hours", "Overtime", "Sick", "Vacation", "Total", None, None],
            4: ["Monday", "11/28/2016", "6.00", None, None, None, "6.00", None, None],
            5: ["Tuesday", "11/29/2016", "6.00", None, None, None, "6.00", None, None],
            6: ["Wednesday", "11/30/2016", "6.00", None, None, None, "6.00", None, None],
            7: ["Thursday", "12/1/2016", "6.00", None, None, None, "6.00", None, None],
            8: ["Friday", "12/2/2016", "6.00", None, None, None, "6.00", None, None],
            9: ["Saturday", None, None, None, None, None, "0.00", None, None],
            10: ["Sunday", None, None, None, None, None, "0.00", None, None],
            11: ["Total hours", None, "30.00", "0.00", "0.00", "0.00", "30.00", None, None],
            12: ["Rate per hour", None, "$60", None, None, None, None, None, None],
            13: ["Total pay", None, "$ 1,800.00", "$ -", "$ -", "$ -", "$1,800.00", None, None]
        }
        # Pandas DataFrame constructor expects data organized by columns for .from_dict,
        # or list of lists for rows. The dict above is column-oriented.
        # For row-oriented like the PDF visual, easier to build list of lists:
        list_of_rows = [
            ["Employee", "Lovina Lalye"],
            ["[Street Address]", "422 Timor Terr"],
            ["[Address 2]", None],
            ["Code]", "sunnyvale, CA, 94089"],
            ["Week ending:", "12/4/2016"],
            [], # Empty row
            ["Manager:", "Ian Gomez"],
            ["phone:", "650-793-1285"],
            ["mail:", "lovina@datatorrent.com"],
            [], # Empty row
            ["Day", "Date", "Regular Hours", "Overtime", "Sick", "Vacation", "Total"],
            ["Monday", "11/28/2016", "6.00", None, None, None, "6.00"],
            ["Tuesday", "11/29/2016", "6.00", None, None, None, "6.00"],
            ["Wednesday", "11/30/2016", "6.00", None, None, None, "6.00"],
            ["Thursday", "12/1/2016", "6.00", None, None, None, "6.00"],
            ["Friday", "12/2/2016", "6.00", None, None, None, "6.00"],
            ["Saturday", None, None, None, None, None, "0.00"],
            ["Sunday", None, None, None, None, None, "0.00"],
            ["Total hours", None, "30.00", "0.00", "0.00", "0.00", "30.00"],
            ["Rate per hour", None, "$60"],
            ["Total pay", None, "$ 1,800.00", "$ -", "$ -", "$ -", "$1,800.00"]
        ]
        dummy_df = pd.DataFrame(list_of_rows)
        dummy_df.to_excel(file_path, sheet_name="Sheet1", index=False, header=False)
        print(f"Dummy Excel file '{file_path}' created for testing.")

        timecard_data = parse_timecard_from_excel(file_path, sheet_name=sheet)

        if timecard_data:
            print("\nExtracted Timecard Metadata (from Excel based on PDF sample):")
            for key, value in timecard_data.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print("Could not parse timecard from Excel.")

    except Exception as e:
        print(f"Error during dummy file creation or parsing: {e}")