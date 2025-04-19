import unittest
from unittest.mock import MagicMock, patch
from ..ap_ar_with_chatgpt_impl import QueryOptimizer, num_tokens_from_string, query_financial_assistant

class MockDoc:
    def __init__(self, content):
        self.page_content = content

class TestQueryOptimizer(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("Invoice A - $500 due 2025-04-10"),
            MockDoc("Invoice B - $1200 due 2025-04-15"),
            MockDoc("Invoice C - $300 due 2025-05-01")
        ]

        self.mock_vectorstore = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever

        self.optimizer = QueryOptimizer(self.mock_vectorstore)

    def test_optimize_query_with_token_limit(self):
        query = "List all invoices due in April"
        result = self.optimizer.optimize_query(query, max_tokens=10)

        self.assertIn("Invoice A", result)
        self.assertNotIn("Invoice B", result)
        self.assertIsInstance(result, str)

    def test_token_count_function(self):
        test_text = "Short test sentence."
        tokens = num_tokens_from_string(test_text)
        self.assertGreater(tokens, 0)

class TestFinancialAssistant(unittest.TestCase):
    def test_query_financial_assistant_with_mocked_chains(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_optimizer_ap.optimize_query.return_value = "Mock AP context"
        mock_optimizer_ar.optimize_query.return_value = "Mock AR context"
        mock_agent.run.return_value = "Mock agent response"

        result = query_financial_assistant(
            query="When is the next AP payment due?",
            mode="ap",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Mock agent response", result)
        mock_optimizer_ap.optimize_query.assert_called_once()
        mock_agent.run.assert_called_once()

if __name__ == "__main__":
    unittest.main()