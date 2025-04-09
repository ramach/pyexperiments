# config.py
import os
from dotenv import load_dotenv

def load_api_key():
    """Loads the OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the .env file.")
    return api_key

# --- Configuration Constants ---
DEFAULT_BASE_MODEL = "gpt-3.5-turbo-0125" # Base model for fine-tuning
DEFAULT_CHAT_MODEL = "gpt-4" # Default model for assistant if no fine-tuned ID provided
FALLBACK_CHAT_MODEL = "gpt-3.5-turbo" # Fallback if default fails

FINETUNE_DATA_PATH = "finetune_data.jsonl" # Path for generated JSONL data
FINE_TUNED_MODEL_ID_STORAGE = "fine_tuned_model_id.txt" # File to store the last model ID

# Add dummy data stores here or load from external files if preferred
# Using dicts here for simplicity, same as before
DUMMY_INTERNAL_DATA = {
    "Unit Alpha": {
        "2025Q1": {"Current Assets": 1200000, "Current Liabilities": 800000, "Inventory": 400000, "Cash": 150000},
        "2024Q4": {"Current Assets": 1150000, "Current Liabilities": 750000, "Inventory": 380000, "Cash": 140000},
    },
    "Unit Beta": {
        "2025Q1": {"Current Assets": 2500000, "Current Liabilities": 1500000, "Inventory": 1000000, "Cash": 300000},
        "2024Q4": {"Current Assets": 2400000, "Current Liabilities": 1600000, "Inventory": 950000, "Cash": 280000},
    }
}

DUMMY_PUBLIC_DATA = {
    "tech industry benchmark": {"Average Current Ratio": 1.8, "Average Quick Ratio": 1.1},
    "manufacturing industry benchmark": {"Average Current Ratio": 2.1, "Average Quick Ratio": 0.9}
}