# retrievers/query_optimizer.py
import tiktoken
from typing import List

class QueryOptimizer:
    def __init__(self, documents: List[str], token_limit: int = 512):
        self.documents = documents
        self.token_limit = token_limit
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def optimize_query(self, query: str) -> str:
        sorted_docs = sorted(
            self.documents,
            key=lambda doc: query.lower() in doc.lower(),
            reverse=True
        )

        selected = []
        total_tokens = 0

        for doc in sorted_docs:
            doc_tokens = self.count_tokens(doc)
            if total_tokens + doc_tokens <= self.token_limit:
                selected.append(doc)
                total_tokens += doc_tokens
            else:
                break

        return "\n".join(selected)