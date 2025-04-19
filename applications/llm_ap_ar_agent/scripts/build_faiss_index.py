# scripts/build_faiss_index.py
from ..vectorstore.faiss_store import load_faiss_index
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    docs_path = os.path.join(base_dir, "test", "mock_docs", "mock_data.txt")
    index_path = os.path.join(base_dir, "vectorstore", "faiss_index")

    index = load_faiss_index(index_path=index_path, docs_path=docs_path)
    print("✅ FAISS index built successfully.")
