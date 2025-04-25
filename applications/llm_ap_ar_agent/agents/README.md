```
Added agents for 
AP/AR and invoice processing

New module for pdf_processing (utils/pdf_utils.py)

Handles pdf documents extracting both metadata and line items
Flow is

extract_data_from_pdf -> data_mapping (to structured data) -> run through lang chain agents

Mapping uses LLM + prompt templates

The agent orchestration is now implemented in agents/invoice_agent.py, llm_ap_ar_agent and llm_invoice_agent.
It wires together all invoice tools and exposes them via OpenAI function calling using LangChain's initialize_agent.
 
Once you checkout the repo
cd llm_ap_ar_agent/applications

Invoke one of these 2 demo UIs to execute a flow

streamlit run ui/invoice_agent_ui2.py (has examples for all types of input and executes invoice chain or AP/AR agent)

for mapping and parsing complex PDFs use

streamlit run ui/llm_invoice_agent_ui.py 
(supports only PDF)

These UIs invoke backend agents (invoice_agent and llm_invoice_agent respectively)
```

```
Extracted raw data from a PDF

Order ID: 10252
Customer ID: SUPRD
Order Date: 2016-07-09
Customer Details:
Contact Name: Pascale Cartrain
Address: Boulevard Tirou, 255
City: Charleroi
Postal Code: B-6000
Country: Belgium
Phone: (071) 23 67 22 20
Fax: (071) 23 67 22 21
Product Details:
Product ID Product Name Quantity Unit Price
20 Sir Rodney's Marmalade 40 64.8
33 Geitost 25 2.0
60 Camembert Pierrot 40 27.2
TotalPrice 3730.0
```
Mapped Data (DEBUG:agents.llm_invoice_agent:[InvoiceAgent_mapping] structured_data)
``` json
{
  "invoice_id": "10252",
  "vendor": "SUPRD",
  "date": "2016-07-09",
  "amount": "3730.0",
  "purchase_order": [
    {
      "product_id": "20",
      "product_name": "Sir Rodney's Marmalade",
      "quantity": "40",
      "unit_price": "64.8"
    },
    {
      "product_id": "33",
      "product_name": "Geitost",
      "quantity": "25",
      "unit_price": "2.0"
    },
    {
      "product_id": "60",
      "product_name": "Camembert Pierrot",
      "quantity": "40",
      "unit_price": "27.2"
    }
  ],
  "payment_method": "MISSING"
}
```
Structured Data Schema
```text
class InvoiceInput(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    date: str
    purchase_order: str
    payment_method: str
```
Mapping Confidence Score:
```text
DEBUG:agents.llm_invoice_agent:[InvoiceAgent_mapping_with_confidence]
 structured_data:
  {'invoice_id': {'value': '10252', 'confidence': 1.0},
  'vendor': {'value': 'SUPRD', 'confidence': 1.0},
   'date': {'value': '2016-07-09', 'confidence': 1.0},
    'amount': {'value': '3730.0', 'confidence': 1.0},
     'purchase_order': {'value': 'MISSING', 'confidence': 0.0},
      'payment_method': {'value': 'MISSING', 'confidence': 0.0}}
```

Sample data for agent execution is available under mockdata

```text
Support for docx
Mapping is enabled for word documents

Detect if file is .docx.

Extract raw text using extract_text_from_docx.

Map using map_docx_text_to_invoice_data.

Then pass to your unified run_llm_invoice_agent(query, input_data) function.
```
