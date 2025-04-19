# tests/test_e2e_chat_agent.py
import unittest
from unittest.mock import patch
from ..agents.openai_function_handler import get_openai_tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

class TestEndToEndChatAgent(unittest.TestCase):

    @patch("langchain.chat_models.ChatOpenAI")
    def test_query_vector_qa_tool(self, mock_llm):
        tools = get_openai_tools()
        llm = mock_llm()
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

        mock_llm().run.return_value = "Vendor TechWave has 2 overdue invoices."

        result = agent.run("Which vendors have overdue invoices?")
        self.assertIn("overdue invoices", result)

    @patch("langchain.chat_models.ChatOpenAI")
    def test_ap_chain_routing(self, mock_llm):
        tools = get_openai_tools()
        agent = initialize_agent(tools, mock_llm(), agent=AgentType.OPENAI_FUNCTIONS)

        mock_llm().run.return_value = "Pending payments found for GreenTech Inc."
        result = agent.run("List pending payments for GreenTech Inc")
        self.assertIn("pending", result.lower())

    @patch("langchain.chat_models.ChatOpenAI")
    def test_public_vendor_lookup(self, mock_llm):
        tools = get_openai_tools()
        agent = initialize_agent(tools, mock_llm(), agent=AgentType.OPENAI_FUNCTIONS)

        mock_llm().run.return_value = "Blue Ocean Systems is a leading supply chain vendor."
        result = agent.run("Tell me about Blue Ocean Systems")
        self.assertIn("Blue Ocean", result)

if __name__ == '__main__':
    unittest.main()
