# vectorstore/json_faiss_builder.py
import os
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def json_to_documents(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    documents = []
    for section in ["accounts_payable", "accounts_receivable"]:
        for entry in data.get(section, []):
            text = " | ".join(f"{k}: {v}" for k, v in entry.items())
            documents.append(Document(page_content=text, metadata={"source": section}))

    # Optionally split text into smaller chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    split_docs = splitter.split_documents(documents)
    return split_docs


def build_faiss_index(json_path: str, index_path: str):
    documents = json_to_documents(json_path)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index built and saved to {index_path}")
    return vectorstore


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    json_file = os.path.join(base_dir, "data", "synthetic", "ap_ar_data.json")
    index_output = os.path.join(base_dir, "vectorstore", "faiss_index")

    build_faiss_index(json_file, index_output)