# main_driver.py
from __future__ import annotations

import argparse
import os
from openai import OpenAI

# Import logic from other modules
import config
import fine_tuning_liquidity_analysis
import liquidity_analysis_assistant

def save_model_id(model_id: str, storage_path: str):
    """Saves the fine-tuned model ID to a file."""
    try:
        with open(storage_path, "w") as f:
            f.write(model_id)
        print(f"Fine-tuned model ID '{model_id}' saved to {storage_path}")
    except IOError as e:
        print(f"Error saving model ID to {storage_path}: {e}")

def load_saved_model_id(storage_path: str) -> str | None:
    """Loads the fine-tuned model ID from a file."""
    if os.path.exists(storage_path):
        try:
            with open(storage_path, "r") as f:
                model_id = f.read().strip()
                if model_id:
                    print(f"Loaded fine-tuned model ID '{model_id}' from {storage_path}")
                    return model_id
        except IOError as e:
            print(f"Error reading model ID from {storage_path}: {e}")
    print(f"No valid model ID found in {storage_path}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Liquidity Analysis Assistant or Fine-Tuning.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["fine-tune", "run-assistant"],
        help="Action to perform: 'fine-tune' or 'run-assistant'."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Specify a fine-tuned model ID to use for the assistant (overrides saved ID)."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="liquidity-v1",
        help="Optional suffix for the fine-tuning job name."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=config.DEFAULT_BASE_MODEL,
        help=f"Base model for fine-tuning (default: {config.DEFAULT_BASE_MODEL})."
    )

    args = parser.parse_args()

    # --- Common Setup ---
    try:
        api_key = config.load_api_key()
        # Initialize OpenAI client (needed for fine-tuning, potentially useful elsewhere)
        client = OpenAI(api_key=api_key)
    except ValueError as e:
        print(e)
        return
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return


    # --- Action Execution ---
    if args.action == "fine-tune":
        print("\n*** Starting Fine-Tuning Action ***")
        fine_tuned_model_id = fine_tuning_liquidity_analysis.run_fine_tuning_pipeline(
            client=client,
            data_file_path=config.FINETUNE_DATA_PATH,
            base_model=args.base_model,
            suffix=args.suffix
        )
        if fine_tuned_model_id:
            save_model_id(fine_tuned_model_id, config.FINE_TUNED_MODEL_ID_STORAGE)
        else:
            print("Fine-tuning did not produce a valid model ID.")

    elif args.action == "run-assistant":
        print("\n*** Starting Assistant Action ***")
        # Determine which model ID to use
        model_id_to_use = args.model_id # Prioritize CLI argument
        if not model_id_to_use:
            # If no CLI arg, try loading saved ID
            model_id_to_use = load_saved_model_id(config.FINE_TUNED_MODEL_ID_STORAGE)
            if not model_id_to_use:
                print("No specific or saved fine-tuned model ID found. Using default model.")

        try:
            # Initialize LLM (handles None model_id_to_use)
            llm = liquidity_analysis_assistant.initialize_llm(api_key, model_id_to_use)

            # Create Tools
            tools = liquidity_analysis_assistant.create_tools()

            # Setup Agent Executor
            agent_executor = liquidity_analysis_assistant.setup_agent_executor(llm, tools)

            # Run Interaction Loop
            liquidity_analysis_assistant.run_assistant_interaction(agent_executor)

        except ConnectionError as e:
            print(f"Assistant setup failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while running the assistant: {e}")

    else:
        print(f"Error: Invalid action '{args.action}' specified.")


if __name__ == "__main__":
    main()