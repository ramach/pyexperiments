import unittest
from unittest.mock import MagicMock
from ..ap_ar_with_chatgpt_impl import QueryOptimizer, num_tokens_from_string

class MockDoc:
    def __init__(self, content):
        self.page_content = content

class TestQueryOptimizer(unittest.TestCase):
    def setUp(self):
        # Mock retriever with dummy documents
        self.mock_retriever = MagicMock()
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("Invoice A - $500 due 2025-04-10"),
            MockDoc("Invoice B - $1200 due 2025-04-15"),
            MockDoc("Invoice C - $300 due 2025-05-01")
        ]

        # Mock vectorstore with as_retriever method
        self.mock_vectorstore = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever

        # Create QueryOptimizer instance with mock vectorstore
        self.optimizer = QueryOptimizer(self.mock_vectorstore)

    def test_optimize_query_with_token_limit(self):
        query = "List all invoices due in April"
        result = self.optimizer.optimize_query(query, max_tokens=10)  # very small token limit

        # Assert only the first document is returned
        self.assertIn("Invoice A", result)
        self.assertNotIn("Invoice B", result)
        self.assertIsInstance(result, str)

    def test_token_count_function(self):
        test_text = "Short test sentence."
        tokens = num_tokens_from_string(test_text)
        self.assertGreater(tokens, 0)

if __name__ == "__main__":
    unittest.main()
