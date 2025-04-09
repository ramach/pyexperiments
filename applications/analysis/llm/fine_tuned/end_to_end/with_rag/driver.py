# main_driver.py
from __future__ import annotations
'''
How to Run (Updated):

Set up project structure, .env file, install libraries (chromadb, langchain-community, pypdf).
Create ./source_documents/ and add your PDF/TXT files.
Build the Index: python main_driver.py --action build-index (Run once initially and when documents change).
(Optional) Fine-tune: python main_driver.py --action fine-tune (Update dummy data in fine_tuning_logic.py first).
Run Assistant: python main_driver.py --action run-assistant (optionally add --model-id ...).
Ask Questions:
Ratio Calculation: "What is the Quick Ratio for Unit Beta in 2024Q4?" (Uses Calculator/Internal Data tools)
Document-based: "Why did Unit Alpha's current assets increase in 2025Q1?" or "Summarize commentary on cash flow risks." (Uses DocumentContextRetriever tool)
Combined: "Calculate Unit Alpha's current ratio for 2025Q1 and explain any significant changes mentioned in reports." (Agent should use multiple tools).
Now your assistant can leverage both structured data lookups/calculations and knowledge retrieved from your indexed documents using ChromaDB and RAG.
Remember to replace dummy implementations and manage security appropriately.
'''
import argparse
import os
from openai import OpenAI

# Import logic from other modules
import config
from analysis.llm.fine_tuned.end_to_end import liquidity_analysis_assistant
from analysis.llm.fine_tuned.end_to_end.with_rag import liquidity_analysis_assistant_rag
from  ..fine_tuning_liquidity_analysis import *
from liquidity_analysis_assistant_rag import * # Make sure assistant_logic has the new RAG functions

# (Keep save_model_id and load_saved_model_id functions as before)
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
        choices=["build-index", "fine-tune", "run-assistant"], # Added build-index
        help="Action to perform: 'build-index', 'fine-tune', or 'run-assistant'."
    )
    # (Keep --model-id, --suffix, --base-model arguments as before)
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Specify a fine-tuned model ID to use for the assistant (overrides saved ID)."
    )
    parser.add_argument(
        "--suffix", type=str, default="liquidity-v1",
        help="Optional suffix for the fine-tuning job name."
    )
    parser.add_argument(
        "--base-model", type=str, default=config.DEFAULT_BASE_MODEL,
        help=f"Base model for fine-tuning (default: {config.DEFAULT_BASE_MODEL})."
    )

    args = parser.parse_args()

    # --- Common Setup ---
    try:
        api_key = config.load_api_key()
        # OpenAI client needed for fine-tuning AND embeddings
        client = OpenAI(api_key=api_key)
    except ValueError as e:
        print(e)
        return
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return


    # --- Action Execution ---
    if args.action == "build-index":
        print("\n*** Starting Vector Store Index Build Action ***")
        # Import and run the build function directly (it uses config for paths)
        from build_vector_store import build_index
        build_index()


    elif args.action == "fine-tune":
        print("\n*** Starting Fine-Tuning Action ***")
        # (Fine-tuning logic remains the same)
        fine_tuned_model_id = run_fine_tuning_pipeline(
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
        print("\n*** Starting Assistant Action (with RAG) ***")
        model_id_to_use = args.model_id or load_saved_model_id(config.FINE_TUNED_MODEL_ID_STORAGE)
        if not model_id_to_use:
            print("No specific or saved fine-tuned model ID found. Using default model.")

        try:
            # Initialize Embeddings (needed for retriever)
            embeddings = liquidity_analysis_assistant_rag.initialize_embeddings(api_key)

            # Get Retriever (loads existing index)
            retriever = liquidity_analysis_assistant_rag.get_vector_store_retriever(
                db_path=config.CHROMA_DB_PATH, # Ensure path is in config
                collection_name=config.COLLECTION_NAME, # Ensure collection name is in config
                embeddings=embeddings
            )

            # Initialize LLM
            llm = liquidity_analysis_assistant_rag.initialize_llm(api_key, model_id_to_use)

            # Create Tools (passing the retriever)
            tools =liquidity_analysis_assistant_rag.create_tools(retriever=retriever)

            # Setup Agent Executor
            agent_executor = liquidity_analysis_assistant_rag.setup_agent_executor(llm, tools)

            # Run Interaction Loop
            liquidity_analysis_assistant_rag.run_assistant_interaction(agent_executor)

        except (ConnectionError, FileNotFoundError) as e:
            print(f"Assistant setup failed: {e}")
            if isinstance(e, FileNotFoundError):
                print("Did you run '--action build-index' first?")
        except Exception as e:
            print(f"An unexpected error occurred while running the assistant: {e}")

    else:
        print(f"Error: Invalid action '{args.action}' specified.")


if __name__ == "__main__":
    main()