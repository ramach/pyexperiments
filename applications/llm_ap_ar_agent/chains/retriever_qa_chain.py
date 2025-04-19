# chains/retriever_qa_chain.py
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os


def load_retriever_chain(index_path: str = None):
    if index_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(base_dir, "vectorstore", "faiss_index")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    return chain


def run_retriever_query(query: str) -> str:
    qa_chain = load_retriever_chain()
    result = qa_chain({"query": query})
    return result["result"]