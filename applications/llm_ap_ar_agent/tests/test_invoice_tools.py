import unittest
from tools import invoice_tools_enhanced

class TestInvoiceTools(unittest.TestCase):

    def test_simulate_po_lookup_found(self):
        result = invoice_tools.simulate_po_lookup("PO-1001")
        self.assertEqual(result, 1500.00)

    def test_simulate_po_lookup_not_found(self):
        result = invoice_tools.simulate_po_lookup("PO-9999")
        self.assertIsNone(result)

    def test_run_invoice_verification_success(self):
        invoice = {
            "invoice_id": "INV-001",
            "vendor": "ACME Inc.",
            "amount": 100.00,
            "date": "2024-01-15"
        }
        result = invoice_tools.run_invoice_verification(invoice)
        self.assertIn("verification passed", result)

    def test_run_invoice_verification_missing_fields(self):
        invoice = {
            "invoice_id": "INV-002",
            "vendor": "ACME Inc."
        }
        result = invoice_tools.run_invoice_verification(invoice)
        self.assertIn("Missing fields", result)

    def test_run_po_matching_success(self):
        invoice = {
            "purchase_order": "PO-1002",
            "amount": 299.99
        }
        result = invoice_tools.run_po_matching(invoice)
        self.assertIn("matches PO", result)

    def test_run_po_matching_mismatch(self):
        invoice = {
            "purchase_order": "PO-1002",
            "amount": 999.99
        }
        result = invoice_tools.run_po_matching(invoice)
        self.assertIn("Amount mismatch", result)

    def test_run_po_matching_not_found(self):
        invoice = {
            "purchase_order": "PO-0000",
            "amount": 50.00
        }
        result = invoice_tools.run_po_matching(invoice)
        self.assertIn("No matching PO", result)

    def test_run_approval_process_single_approver(self):
        invoice = {"amount": 500.00}
        result = invoice_tools.run_approval_process(invoice)
        self.assertIn("manager@company.com", result)
        self.assertNotIn("finance@company.com", result)

    def test_run_approval_process_multiple_approvers(self):
        invoice = {"amount": 2000.00}
        result = invoice_tools.run_approval_process(invoice)
        self.assertIn("manager@company.com", result)
        self.assertIn("finance@company.com", result)

    def test_run_payment_processing(self):
        invoice = {
            "vendor": "ACME Inc.",
            "amount": 750.00,
            "payment_method": "credit_card"
        }
        result = invoice_tools.run_payment_processing(invoice)
        self.assertIn("Scheduled credit_card payment", result)

if __name__ == '__main__':
    unittest.main()
