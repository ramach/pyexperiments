from typing import List
import tiktoken

class QueryOptimizer:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def optimize_query(self, query: str, max_tokens: int = 500) -> str:
        retriever = self.vectorstore.as_retriever()
        documents = retriever.get_relevant_documents(query)

        selected_docs = []
        current_tokens = 0

        for doc in documents:
            doc_text = doc.page_content.strip()
            doc_tokens = num_tokens_from_string(doc_text)

            if current_tokens + doc_tokens > max_tokens:
                break

            selected_docs.append(doc_text)
            current_tokens += doc_tokens

        return "\n---\n".join(selected_docs)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def query_financial_assistant(query: str, mode: str, qa_chain_ap, qa_chain_ar,
                              query_optimizer_ap, query_optimizer_ar, agent):
    if mode == "ap":
        context = query_optimizer_ap.optimize_query(query)
        return agent.run(context)
    elif mode == "ar":
        context = query_optimizer_ar.optimize_query(query)
        return agent.run(context)
    elif mode == "vendor":
        return agent.run(query)
    elif mode == "public":
        return agent.run(query)
    else:
        return "Invalid mode: must be one of ['ap', 'ar', 'vendor', 'public']"