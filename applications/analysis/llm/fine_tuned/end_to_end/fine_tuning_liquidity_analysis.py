from __future__ import annotations

import json
import time
import os
from openai import OpenAI # Make sure openai library is >= 1.0

# --- Data Preparation ---

def get_fine_tuning_examples():
    """Returns a list of dummy fine-tuning examples.
       Replace this with loading your actual high-quality data."""
    # DUMMY Example Data - REPLACE WITH YOUR ACTUAL, HIGH-QUALITY DATA
    return [
        { "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "What is the Current Ratio for Unit Alpha in 2025Q1?"},
            {"role": "assistant", "content": "Okay, to calculate the Current Ratio for Unit Alpha in 2025Q1, I need 'Current Assets' and 'Current Liabilities' for that unit and period. I will use the InternalFinancialDataLookup tool.\n\nAction: InternalFinancialDataLookup(query='Get Current Assets for Unit Alpha for 2025Q1')\nObservation: {'Current Assets': 1200000}\nAction: InternalFinancialDataLookup(query='Get Current Liabilities for Unit Alpha for 2025Q1')\nObservation: {'Current Liabilities': 800000}\nAction: LiquidityRatioCalculator(input_data={'ratio_to_calculate': 'Current Ratio', 'Current Assets': 1200000, 'Current Liabilities': 800000})\nObservation: Calculated Current Ratio: 1.50\n\nFinal Answer: The Current Ratio for Unit Alpha in 2025Q1 is 1.50."}
        ]},
        { "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "How does Unit Beta's Quick Ratio of 0.75 in 2024Q4 look?"},
            {"role": "assistant", "content": "Unit Beta's Quick Ratio of 0.75 in 2024Q4 indicates its liquid assets (excluding inventory) cover 75% of its current liabilities. For Unit Beta, which is inventory-heavy, this is below our internal target of 0.9 but slightly improved from 0.72 in Q3. We should monitor its cash flow closely."}
        ]},
        { "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "Calculate quick ratio unit alpha 2025Q1"},
            {"role": "assistant", "content": "Understood. To calculate the Quick Ratio for Unit Alpha in 2025Q1, I need 'Current Assets', 'Inventory', and 'Current Liabilities' for that period. I'll use the internal lookup tool.\n\nAction: InternalFinancialDataLookup(query='Get Current Assets for Unit Alpha for 2025Q1')\nObservation: {'Current Assets': 1200000}\nAction: InternalFinancialDataLookup(query='Get Inventory for Unit Alpha for 2025Q1')\nObservation: {'Inventory': 400000}\nAction: InternalFinancialDataLookup(query='Get Current Liabilities for Unit Alpha for 2025Q1')\nObservation: {'Current Liabilities': 800000}\nAction: LiquidityRatioCalculator(input_data={'ratio_to_calculate': 'Quick Ratio', 'Current Assets': 1200000, 'Inventory': 400000, 'Current Liabilities': 800000})\nObservation: Calculated Quick Ratio: 1.00\n\nFinal Answer: The Quick Ratio for Unit Alpha in 2025Q1 is 1.00."}
        ]},
        # !!! ADD MANY MORE HIGH-QUALITY EXAMPLES HERE !!!
    ]

def prepare_fine_tuning_data(examples: list, output_path: str) -> bool:
    """Converts examples to JSONL format and writes to file."""
    try:
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"Data successfully written to {output_path}")
        return True
    except IOError as e:
        print(f"Error writing data to {output_path}: {e}")
        return False

# --- OpenAI API Interactions ---
def upload_file_to_openai(client: OpenAI, file_path: str) -> str | None:
    """Uploads a file to OpenAI for fine-tuning."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        print(f"Uploading {file_path} to OpenAI...")
        with open(file_path, "rb") as f:
            training_file = client.files.create(file=f, purpose="fine-tune")
        print(f"File uploaded successfully. File ID: {training_file.id}")
        return training_file.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def create_fine_tuning_job(client: OpenAI, training_file_id: str, base_model: str, suffix: str = None) -> str | None:
    """Creates the fine-tuning job."""
    try:
        print(f"Creating fine-tuning job using model '{base_model}' and file '{training_file_id}'...")
        job_payload = {
            "training_file": training_file_id,
            "model": base_model,
        }
        if suffix:
            job_payload["suffix"] = suffix

        job = client.fine_tuning.jobs.create(**job_payload)
        job_id = job.id
        print(f"Fine-tuning job created successfully. Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

def monitor_fine_tuning_job(client: OpenAI, job_id: str) -> tuple[str | None, str | None]:
    """Monitors job, returns (status, fine_tuned_model_id)."""
    print("\nMonitoring fine-tuning job status (Checks every 60s, press Ctrl+C to interrupt)...")
    final_status = None
    fine_tuned_model_id = None
    try:
        while True:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Job Status: {status}")

            if status in ["succeeded", "failed", "cancelled"]:
                final_status = status
                print(f"\nJob finished with status: {final_status}")
                if status == "succeeded":
                    fine_tuned_model_id = job_status.fine_tuned_model
                    print(f"Fine-tuned Model ID: {fine_tuned_model_id}")
                else:
                    print(f"Job failed or was cancelled. Details: {job_status}")
                break # Exit the loop
            time.sleep(60) # Wait before checking again
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
        final_status = "interrupted"
        # Optionally try to cancel job here if desired
    except Exception as e:
        print(f"\nAn error occurred during monitoring: {e}")
        final_status = "error_during_monitoring"

    return final_status, fine_tuned_model_id

def run_fine_tuning_pipeline(client: OpenAI, data_file_path: str, base_model: str, suffix: str = "liquidity-v1") -> str | None:
    """Orchestrates the fine-tuning process."""
    print("--- Starting Fine-Tuning Pipeline ---")

    # 1. Prepare Data (using dummy data getter here)
    examples = get_fine_tuning_examples()
    if not examples:
        print("Error: No fine-tuning examples provided.")
        return None
    if not prepare_fine_tuning_data(examples, data_file_path):
        return None

    # 2. Upload File
    training_file_id = upload_file_to_openai(client, data_file_path)
    if not training_file_id:
        return None

    # 3. Create Job
    job_id = create_fine_tuning_job(client, training_file_id, base_model, suffix)
    if not job_id:
        # Consider deleting uploaded file on immediate failure
        # client.files.delete(training_file_id)
        return None

    # 4. Monitor Job
    status, fine_tuned_model_id = monitor_fine_tuning_job(client, job_id)

    # 5. Optional: Clean up uploaded file locally if desired (OpenAI keeps a copy)
    # if os.path.exists(data_file_path):
    #     os.remove(data_file_path)
    #     print(f"Removed local data file: {data_file_path}")
    # Consider deleting the file on OpenAI platform if no longer needed
    # try: client.files.delete(training_file_id) except: pass


    if status == "succeeded" and fine_tuned_model_id:
        print("--- Fine-Tuning Pipeline Completed Successfully ---")
        return fine_tuned_model_id
    else:
        print("--- Fine-Tuning Pipeline Did Not Complete Successfully ---")
        return None