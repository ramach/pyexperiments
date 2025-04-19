# tests/test_invoice_agent.py
import sys
import os
#Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import json
from agents.invoice_agent import run_invoice_agent_with_function_calling

class TestInvoiceAgent(unittest.TestCase):

    def setUp(self):
        with open("mockdata/invoices/invoice2.json") as f:
            self.valid_invoice = json.load(f)

        with open("mockdata/invoice_invalid.json") as f:
            self.missing_fields_invoice = json.load(f)

    def test_valid_invoice(self):
        input_str = json.dumps(self.valid_invoice)
        response = run_invoice_agent_with_function_calling("Is this invoice valid?", input_str)
        self.assertIn("verification passed", response)
        #self.assertIn("approval granted", response.lower())
        #self.assertIn("payment of", response.lower())

    def test_missing_fields(self):
        input_str = json.dumps(self.missing_fields_invoice)
        response = run_invoice_agent_with_function_calling(input_str)
        self.assertIn("missing fields", response.lower())

    def test_invalid_input(self):
        response = run_invoice_agent_with_function_calling("Not a JSON string")
        self.assertIn("error", response.lower())

if __name__ == '__main__':
    unittest.main()
