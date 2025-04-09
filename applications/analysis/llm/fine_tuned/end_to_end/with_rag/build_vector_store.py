# build_vector_store.py
import os
import shutil
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

import config # Assuming config.py has load_api_key() and defines storage path

# --- Configuration ---
# Ensure these paths are correct and exist
SOURCE_DOCUMENTS_PATH = "./source_documents" # Create this folder and put PDFs/TXTs inside
CHROMA_DB_PATH = "./chroma_db_liquidity"
COLLECTION_NAME = "liquidity_docs"

# Create source documents directory if it doesn't exist
if not os.path.exists(SOURCE_DOCUMENTS_PATH):
    os.makedirs(SOURCE_DOCUMENTS_PATH)
    print(f"Created directory {SOURCE_DOCUMENTS_PATH}. Please add your source documents (PDF, TXT) there.")

def build_index():
    """Loads documents, splits them, embeds them, and stores them in ChromaDB."""
    print("--- Starting Vector Store Index Build ---")

    # Clean up old database directory if it exists
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Removing existing database at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)

    # Load API Key for Embeddings
    try:
        api_key = config.load_api_key()
    except ValueError as e:
        print(e)
        return

    # Check if source documents directory is empty
    if not os.listdir(SOURCE_DOCUMENTS_PATH):
        print(f"Error: Source documents directory '{SOURCE_DOCUMENTS_PATH}' is empty.")
        print("Please add PDF or TXT files to this directory before running the build.")
        return

    # Load Documents (using DirectoryLoader for simplicity)
    # Supports loading various file types based on glob patterns
    # Ensure you have necessary loaders installed (e.g., pypdf for PDFs)
    print(f"Loading documents from {SOURCE_DOCUMENTS_PATH}...")
    # Adjust glob pattern if needed, e.g., add "*.docx" and install python-docx
    loader = DirectoryLoader(SOURCE_DOCUMENTS_PATH, glob="**/*[!.gitkeep]", show_progress=True, use_multithreading=True)
    try:
        documents = loader.load()
        if not documents:
            print("Error: No documents were loaded. Check paths and file types.")
            return
        print(f"Loaded {len(documents)} document sections.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # Split Documents into Chunks
    # Adjust chunk_size and chunk_overlap as needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    if not texts:
        print("Error: No text chunks generated after splitting.")
        return

    # Initialize Embedding Model
    print("Initializing embedding model...")
    try:
        # Using OpenAI embeddings, ensure API key is valid
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        return

    # Create ChromaDB Vector Store and Persist
    print(f"Creating/Populating ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DB_PATH}...")
    try:
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH
        )
        # Ensure persistence
        vector_store.persist()
        print("--- Vector Store Index Build Complete ---")
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")


if __name__ == "__main__":
    build_index()