import unittest
from chains.ar_chain_scaffolding.ar_chain import ARChain

class TestARChain(unittest.TestCase):
    def setUp(self):
        self.chain = ARChain()

    def test_receivable_invoice_standard(self):
        invoice = {
            "customer": "Beta Co",
            "invoice_id": "AR-2025-01",
            "amount": 4800.0,
            "due_date": "2025-04-20"
        }
        result = self.chain.run(invoice)
        self.assertIn("payment", result.lower())
        self.assertTrue("follow" in result.lower())

    def test_late_invoice(self):
        invoice = {
            "customer": "Zeta Ltd",
            "invoice_id": "AR-2024-89",
            "amount": 12000.0,
            "due_date": "2025-03-10"
        }
        result = self.chain.run(invoice)
        self.assertIn("delayed", result.lower())

if __name__ == "__main__":
    unittest.main()