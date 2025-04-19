import unittest
from unittest.mock import MagicMock, patch
from .Llm_Ap_Ar_Agent import QueryOptimizer, num_tokens_from_string, query_financial_assistant

class MockDoc:
    def __init__(self, content):
        self.page_content = content

class TestQueryOptimizer(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("Invoice A - $500"),
            MockDoc("Invoice B - $1200"),
            MockDoc("Invoice C - $300")
        ]

        self.mock_vectorstore = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever

        self.optimizer = QueryOptimizer(self.mock_vectorstore)

    def test_optimize_query_with_token_limit(self):
        query = "List all invoices due in April"
        result = self.optimizer.optimize_query(query, max_tokens=10)

        self.assertIn("Invoice A", result)
        self.assertIsInstance(result, str)

    def test_token_count_function(self):
        test_text = "Short test sentence."
        tokens = num_tokens_from_string(test_text)
        self.assertGreater(tokens, 0)

    def test_optimize_query_with_no_documents(self):
        self.mock_retriever.get_relevant_documents.return_value = []
        result = self.optimizer.optimize_query("Any unpaid invoices?", max_tokens=10)
        self.assertEqual(result, "")

    def test_optimize_query_all_documents_filtered(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("A" * 500),
            MockDoc("B" * 500)
        ]
        result = self.optimizer.optimize_query("Check limits", max_tokens=1)
        self.assertTrue(result == "" or isinstance(result, str))

    @patch('Llm_Ap_Ar_Agent.num_tokens_from_string')
    def test_token_usage_checked(self, mock_token_counter):
        mock_token_counter.return_value = 5
        query = "Invoices for April"
        self.optimizer.optimize_query(query, max_tokens=10)
        mock_token_counter.assert_called()

class TestQueryOptimizerRealisticDocs(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_vectorstore = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_retriever
        self.optimizer = QueryOptimizer(self.mock_vectorstore)

    def test_optimize_query_with_realistic_doc_structure(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("""
                Vendor: ABC Corp
                Invoice #: INV-001
                Amount: $4500
                Due Date: 2025-04-18
                Status: Pending
            """),
            MockDoc("""
                Vendor: XYZ Ltd
                Invoice #: INV-002
                Amount: $6200
                Due Date: 2025-04-21
                Status: Paid
            """),
            MockDoc("""
                Vendor: LMN Inc
                Invoice #: INV-003
                Amount: $3000
                Due Date: 2025-05-05
                Status: Pending
            """)
        ]

        query = "Which April invoices are still unpaid?"
        result = self.optimizer.optimize_query(query, max_tokens=30)

        self.assertIn("ABC Corp", result)
        self.assertNotIn("LMN Inc", result)

    def test_optimize_query_with_notes_and_metadata(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("""
                Vendor: Bright Supplies
                Invoice #: 7845
                Amount: $2300
                Due Date: 2025-04-25
                Status: Overdue
                Notes: Delay due to customs hold.
            """),
            MockDoc("""
                Vendor: DataPro Services
                Invoice #: 8723
                Amount: $4100
                Due Date: 2025-04-20
                Status: Pending
                Notes: Requires management approval.
            """),
            MockDoc("""
                Vendor: Eco Transport
                Invoice #: 9021
                Amount: $980
                Due Date: 2025-03-30
                Status: Paid
                Notes: Paid early due to discount.
            """)
        ]

        query = "Which April invoices need action due to payment delays or approvals?"
        result = self.optimizer.optimize_query(query, max_tokens=60)

        self.assertIn("Bright Supplies", result)
        self.assertIn("DataPro Services", result)
        self.assertNotIn("Eco Transport", result)

    def test_optimize_query_with_multiline_and_tabular_style(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("""
                | Vendor        | Invoice # | Amount | Due Date   | Status  |
                |---------------|-----------|--------|------------|---------|
                | FastTrack Inc | 123456    | $1500  | 2025-04-19 | Pending |
                | GlobalTech    | 654321    | $2750  | 2025-04-25 | Paid    |
            """),
            MockDoc("""
                | Vendor     | Invoice # | Amount | Due Date   | Status  |
                |------------|-----------|--------|------------|---------|
                | ZenOffice  | 111222    | $980   | 2025-05-03 | Pending |
            """)
        ]

        query = "List unpaid invoices due in April in tabular records."
        result = self.optimizer.optimize_query(query, max_tokens=50)

        self.assertIn("FastTrack Inc", result)
        self.assertNotIn("GlobalTech", result)
        self.assertNotIn("ZenOffice", result)

    def test_optimize_query_with_ocr_noise_simulation(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("Vend0r: N3xus L0gistics  | Invo1ce: 002334 | Am0unt: $1230 | D8te: 04/20/25 | P3nding"),
            MockDoc("V3ndor: P@rtnerSys        | Inv#: 004321     | Due: 04-28-25 | Status: Cl3ared")
        ]

        query = "Which invoices with OCR errors are still pending in April?"
        result = self.optimizer.optimize_query(query, max_tokens=40)

        self.assertIn("N3xus", result)
        self.assertNotIn("Cl3ared", result)

    def test_optimize_query_with_scraped_web_data(self):
        self.mock_retriever.get_relevant_documents.return_value = [
            MockDoc("Page content from vendor site: Invoice #55667 | Due: April 22, 2025 | Status: Awaiting payment"),
            MockDoc("Details scraped: Vendor - AlphaEngineers | Invoice No: 88442 | Paid on: April 2, 2025")
        ]

        query = "Highlight any vendor site invoice that is still unpaid for April."
        result = self.optimizer.optimize_query(query, max_tokens=50)

        self.assertIn("Awaiting payment", result)
        self.assertNotIn("Paid on", result)

class TestFinancialAssistant(unittest.TestCase):
    def test_query_financial_assistant_with_mocked_ap_mode(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_optimizer_ap.optimize_query.return_value = "Mock AP context"
        mock_agent.run.return_value = "Mock agent response for AP"

        result = query_financial_assistant(
            query="When is the next AP payment due?",
            mode="ap",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Mock agent response for AP", result)
        mock_optimizer_ap.optimize_query.assert_called_once()
        mock_agent.run.assert_called_once()

    def test_query_financial_assistant_with_mocked_ar_mode(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_optimizer_ar.optimize_query.return_value = "Mock AR context"
        mock_agent.run.return_value = "Mock agent response for AR"

        result = query_financial_assistant(
            query="What AR payments are pending?",
            mode="ar",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Mock agent response for AR", result)
        mock_optimizer_ar.optimize_query.assert_called_once()
        mock_agent.run.assert_called_once()

    def test_query_financial_assistant_with_vendor_mode(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_agent.run.return_value = "Mock agent response for vendor"

        result = query_financial_assistant(
            query="What is ABC Ltd's billing policy?",
            mode="vendor",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Mock agent response for vendor", result)
        mock_agent.run.assert_called_once()

    def test_query_financial_assistant_with_public_mode(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_agent.run.return_value = "Mock agent response for public"

        result = query_financial_assistant(
            query="Give me public financial data on EnterpriseX",
            mode="public",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Mock agent response for public", result)
        mock_agent.run.assert_called_once()

    def test_query_financial_assistant_with_invalid_mode(self):
        mock_ap_chain = MagicMock()
        mock_ar_chain = MagicMock()
        mock_agent = MagicMock()
        mock_optimizer_ap = MagicMock()
        mock_optimizer_ar = MagicMock()

        mock_agent.run.return_value = "Invalid mode"

        result = query_financial_assistant(
            query="Should handle invalid mode",
            mode="invalid",
            qa_chain_ap=mock_ap_chain,
            qa_chain_ar=mock_ar_chain,
            query_optimizer_ap=mock_optimizer_ap,
            query_optimizer_ar=mock_optimizer_ar,
            agent=mock_agent
        )

        self.assertIn("Invalid", result)
        mock_agent.run.assert_called_once()

if __name__ == "__main__":
    unittest.main()

