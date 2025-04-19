# vectorstore/faiss_store.py
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

def load_faiss_index(index_path: str, docs_path: str = None) -> FAISS:
    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)

    if docs_path is None:
        raise ValueError("Index not found and no docs_path provided to create new index.")

    loader = TextLoader(docs_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    index = FAISS.from_documents(split_docs, embeddings)
    index.save_local(index_path)
    return index