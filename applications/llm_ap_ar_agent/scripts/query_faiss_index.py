# scripts/query_faiss_index.py
from ..vectorstore.faiss_store import load_faiss_index
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Query to search the FAISS index")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    index_path = os.path.join(base_dir, "vectorstore", "faiss_index")

    # Load index and QA chain
    index = load_faiss_index(index_path)
    retriever = index.as_retriever()
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")

    # Run query
    docs = retriever.get_relevant_documents(args.query)
    response = chain.run(input_documents=docs, question=args.query)

    print("\nResponse:\n", response)
