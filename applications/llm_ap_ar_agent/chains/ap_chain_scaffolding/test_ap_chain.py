import unittest
from chains.ap_chain_scaffolding.ap_chain import APChain

class TestAPChain(unittest.TestCase):
    def setUp(self):
        self.chain = APChain()

    def test_valid_invoice(self):
        invoice = {
            "vendor": "Acme Corp",
            "invoice_id": "INV-12345",
            "amount": 2500.0,
            "due_date": "2025-04-30"
        }
        result = self.chain.run(invoice)
        self.assertIn("valid", result.lower())
        self.assertTrue(any(keyword in result.lower() for keyword in ["pay", "flag", "follow-up"]))

    def test_high_value_invoice(self):
        invoice = {
            "vendor": "Globex Inc",
            "invoice_id": "INV-98765",
            "amount": 25000.0,
            "due_date": "2025-04-15"
        }
        result = self.chain.run(invoice)
        self.assertIn("flag", result.lower())

if __name__ == "__main__":
    unittest.main()