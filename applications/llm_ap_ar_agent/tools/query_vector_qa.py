# tools/query_vector_qa.py
import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

VECTOR_STORE_PATH = "faiss_store/index.pkl"
EMBEDDING_MODEL = OpenAIEmbeddings()

def seed_faiss_index_if_needed():
    if not os.path.exists("faiss_store"):
        os.makedirs("faiss_store")
    if not os.path.exists(VECTOR_STORE_PATH):
        dummy_doc = Document(page_content="hello world", metadata={"source": "init"})
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = splitter.split_documents([dummy_doc])
        db = FAISS.from_documents(docs, EMBEDDING_MODEL)
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(db, f)

def ingest_pdf_and_index(pdf_file, file_name="uploaded.pdf"):
    reader = PdfReader(pdf_file)
    pages = [
        {"text": page.extract_text(), "source": f"{file_name} - page {i+1}"}
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]
    load_documents_into_vector_store(pages)

def load_documents_into_vector_store(docs):
    if not os.path.exists(VECTOR_STORE_PATH):
        seed_faiss_index_if_needed()

    with open(VECTOR_STORE_PATH, "rb") as f:
        db = pickle.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    langchain_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in docs
    ]
    chunks = splitter.split_documents(langchain_docs)
    db.add_documents(chunks)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(db, f)

def query_vector_similarity_search(query: str, k=5):
    with open(VECTOR_STORE_PATH, "rb") as f:
        db = pickle.load(f)
    results = db.similarity_search(query, k=k)
    return "\n\n".join([f"{doc.metadata['source']}:\n{doc.page_content}" for doc in results])
