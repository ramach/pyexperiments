# test/test_main_and_scraper.py
import unittest
from unittest.mock import patch
from ..agents.llm_ap_ar_agent import run_agent_query, create_financial_agent
from ..retrievers.serpapi_scraper import run_serpapi_scraper

class TestMainAndScraper(unittest.TestCase):

    @patch("agents.llm_ap_ar_agent.run_agent_query")
    def test_main_query_execution(self, mock_run_query):
        mock_run_query.return_value = "AP data processed."
        agent = create_financial_agent()
        result = run_agent_query(agent, "When is the next AP due?")
        self.assertEqual(result, "AP data processed.")

    @patch("retrievers.serpapi_scraper.GoogleSearch.get_dict")
    @patch("retrievers.serpapi_scraper.SERPAPI_KEY", new="fake_key")
    def test_serpapi_scraper_returns_results(self, mock_get_dict):
        mock_get_dict.return_value = {
            "organic_results": [
                {"snippet": "Vendor Z offers cloud billing systems."},
                {"snippet": "Vendor Z API supports real-time invoice tracking."},
            ]
        }
        results = run_serpapi_scraper("Vendor Z billing system")
        self.assertIn("Vendor Z offers cloud billing systems.", results)
        self.assertIn("Vendor Z API supports real-time invoice tracking.", results)

    @patch("retrievers.serpapi_scraper.SERPAPI_KEY", new=None)
    def test_serpapi_key_missing(self):
        result = run_serpapi_scraper("test query")
        self.assertEqual(result, "SERPAPI key is missing.")

if __name__ == '__main__':
    unittest.main()
