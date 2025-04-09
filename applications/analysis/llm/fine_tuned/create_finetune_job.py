import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI # Use the latest OpenAI library version (>= 1.0)

# --- Environment Setup ---
load_dotenv() # Load environment variables from README file

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the README file.")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# --- 1. Prepare Your Fine-tuning Data ---
# This is the MOST IMPORTANT step. Create high-quality examples.
# For chat models, the format is a list of messages per example.
# Aim for at least 50-100 high-quality examples, potentially more.

# DUMMY Example Data - REPLACE WITH YOUR ACTUAL, HIGH-QUALITY DATA
# Focus on teaching desired reasoning, internal jargon, interpretation context.
fine_tuning_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "What is the Current Ratio for Unit Alpha in 2025Q1?"},
            {"role": "assistant", "content": "Okay, to calculate the Current Ratio for Unit Alpha in 2025Q1, I need 'Current Assets' and 'Current Liabilities' for that unit and period. I will use the InternalFinancialDataLookup tool.\n\nAction: InternalFinancialDataLookup(query='Get Current Assets for Unit Alpha for 2025Q1')\nObservation: {'Current Assets': 1200000}\nAction: InternalFinancialDataLookup(query='Get Current Liabilities for Unit Alpha for 2025Q1')\nObservation: {'Current Liabilities': 800000}\nAction: LiquidityRatioCalculator(input_data={'ratio_to_calculate': 'Current Ratio', 'Current Assets': 1200000, 'Current Liabilities': 800000})\nObservation: Calculated Current Ratio: 1.50\n\nFinal Answer: The Current Ratio for Unit Alpha in 2025Q1 is 1.50."} # Example showing tool use reasoning
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "How does Unit Beta's Quick Ratio of 0.75 in 2024Q4 look?"},
            {"role": "assistant", "content": "Unit Beta's Quick Ratio of 0.75 in 2024Q4 indicates its liquid assets (excluding inventory) cover 75% of its current liabilities. For Unit Beta, which is inventory-heavy, this is below our internal target of 0.9 but slightly improved from 0.72 in Q3. We should monitor its cash flow closely."}, # Example showing contextual interpretation
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a financial analyst assistant specializing in liquidity analysis for internal company units."},
            {"role": "user", "content": "Calculate quick ratio unit alpha 2025Q1"}, # Example with less formal query
            {"role": "assistant", "content": "Understood. To calculate the Quick Ratio for Unit Alpha in 2025Q1, I need 'Current Assets', 'Inventory', and 'Current Liabilities' for that period. I'll use the internal lookup tool.\n\nAction: InternalFinancialDataLookup(query='Get Current Assets for Unit Alpha for 2025Q1')\nObservation: {'Current Assets': 1200000}\nAction: InternalFinancialDataLookup(query='Get Inventory for Unit Alpha for 2025Q1')\nObservation: {'Inventory': 400000}\nAction: InternalFinancialDataLookup(query='Get Current Liabilities for Unit Alpha for 2025Q1')\nObservation: {'Current Liabilities': 800000}\nAction: LiquidityRatioCalculator(input_data={'ratio_to_calculate': 'Quick Ratio', 'Current Assets': 1200000, 'Inventory': 400000, 'Current Liabilities': 800000})\nObservation: Calculated Quick Ratio: 1.00\n\nFinal Answer: The Quick Ratio for Unit Alpha in 2025Q1 is 1.00."}
        ]
    },
    # !!! ADD MANY MORE HIGH-QUALITY EXAMPLES HERE !!!
]

# Define the output file path
jsonl_file_path = "finetune_data.jsonl"

# Convert examples to JSONL format and write to file
print(f"Preparing data and writing to {jsonl_file_path}...")
with open(jsonl_file_path, "w") as f:
    for example in fine_tuning_examples:
        f.write(json.dumps(example) + "\n")
print("Data preparation complete.")

# --- 2. Upload the File to OpenAI ---
print(f"\nUploading {jsonl_file_path} to OpenAI...")
try:
    training_file = client.files.create(
        file=open(jsonl_file_path, "rb"), # Open in binary read mode
        purpose="fine-tune"
    )
    print(f"File uploaded successfully. File ID: {training_file.id}")
except Exception as e:
    print(f"Error uploading file: {e}")
    exit() # Exit if upload fails

# --- 3. Create the Fine-tuning Job ---
# Choose the base model to fine-tune. Check OpenAI docs for available models.
# 'gpt-3.5-turbo-0125' is a common choice as of early 2024/2025.
base_model = "gpt-3.5-turbo-0125"
print(f"\nCreating fine-tuning job using model '{base_model}' and file '{training_file.id}'...")

try:
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=base_model,
        # You can optionally add hyperparameters, validation file, suffix etc.
        # suffix="liquidity-v1", # Optional: Adds a custom name suffix to your model
    )
    job_id = job.id
    print(f"Fine-tuning job created successfully. Job ID: {job_id}")
except Exception as e:
    print(f"Error creating fine-tuning job: {e}")
    # Consider deleting the uploaded file if job creation fails immediately
    # client.files.delete(training_file.id)
    exit()

# --- 4. Monitor the Fine-tuning Job ---
print("\nMonitoring fine-tuning job status (Press Ctrl+C to interrupt)...")
try:
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        status = job_status.status
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Job Status: {status}")

        if status in ["succeeded", "failed", "cancelled"]:
            print(f"\nJob finished with status: {status}")
            if status == "succeeded":
                fine_tuned_model_id = job_status.fine_tuned_model
                print(f"Fine-tuned Model ID: {fine_tuned_model_id}")
                print("\nUse this Model ID in your Langchain application!")
            else:
                print(f"Job failed or was cancelled. Details: {job_status}")
            break # Exit the loop

        # Wait before checking again (e.g., every 60 seconds)
        time.sleep(60)

except KeyboardInterrupt:
    print("\nMonitoring interrupted by user.")
    # Optionally offer to cancel the job if needed
    # cancel_job = input("Do you want to try cancelling the job? (y/n): ")
    # if cancel_job.lower() == 'y':
    #     try:
    #         client.fine_tuning.jobs.cancel(job_id)
    #         print(f"Attempted to cancel job {job_id}")
    #     except Exception as e:
    #         print(f"Error cancelling job: {e}")

except Exception as e:
    print(f"\nAn error occurred during monitoring: {e}")

# --- 5. Optional: Clean up uploaded file ---
# print("\nCleaning up uploaded file...")
# try:
#     client.files.delete(training_file.id)
#     print(f"Successfully deleted file {training_file.id}")
# except Exception as e:
#     print(f"Error deleting file {training_file.id}: {e}")

print("\nFine-tuning script finished.")