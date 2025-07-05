```bash

streamlit run ui/streamlit_app.py
```

```txt
Embeddings Generation:

When documents (e.g. invoices, vendor statements) are loaded, their text content is converted into vector embeddings using an embedding model (like OpenAIEmbeddings or HuggingFaceEmbeddings).

Vector Store (FAISS/Chroma):

These embeddings are stored in a vector database like FAISS or Chroma for fast similarity-based retrieval.

Query Embedding and Retrieval:

When the user asks a question, the query is also embedded, and a semantic similarity search is performed against the stored document vectors to find the most relevant pieces of context.

Tool Usage:

Your tool (likely one returned from get_openai_tools()) retrieves relevant document chunks and feeds them into the LLM for reasoning/answering.

```
```txt
✅ Where To Use Embeddings in streamlit_ui.py
Right now, run_agent() just passes the full document context and query to the agent. To use vector store-based retrieval:

When loading PDFs, mock data, or image OCR:

Instead of passing the entire content into run_agent, first embed and store them into FAISS/Chroma.

During question answering:

Embed the query → do similarity search → pass retrieved chunks to the agent.

```
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def build_vector_store(text_data):
    docs = text_splitter.create_documents([text_data])
    return FAISS.from_documents(docs, embedding_model)

def search_vector_store(vector_store, query):
    relevant_docs = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in relevant_docs])
#Then replace this in your "Run Agent" section:

vs = build_vector_store(document_data)
top_docs = search_vector_store(vs, query)
response = run_agent(f"{query}\nUse this context:\n{top_docs}")
#🔌 Bonus Tip: Persisting FAISS
#To avoid re-embedding every time:

#python
#Copy
#Edit
#vector_store.save_local("vector_store/")
# Later:
#vector_store = FAISS.load_local("vector_store/", embedding_model)
```
```txt
✅ Streamlit_embedding UI now supports using vendor-specific public info as an added context to the embedding-based retrieval. This strengthens query relevance and improves the agent’s response quality.
```
```txt
✅ Enhanced the Streamlit UI to support additional AP tasks:

Invoice Verification

PO Matching

Approval Process

Payment Processing

Each mode will prompt the user accordingly and still uses embeddings + context-aware querying to analyze invoice data. Let me know if you’d like to scaffold separate chains or tools for each of these tasks!
```

```
Enhanced the Streamlit UI to support dedicated tasks for:

Invoice Verification

PO Matching

Approval Process

Payment Processing

Each task delegates to a corresponding function (e.g., run_invoice_verification, run_po_matching, etc.) inside a new module: tools/invoice_tools.py.

```
```
The invoice_agent_ui.py file is now updated to include:

Multi-agent routing.

PDF upload and text extraction.

SerpAPI integration for vendor-specific public data.

SQLite integration to simulate a backend database for storing chat history.

Form UI for invoice data.

Chat history panel.
```
```
The Streamlit UI has been enhanced with:

Multi-agent routing support (Invoice, AP, AR, Similarity Search)

Vector similarity search using FAISS

Vendor data retrieval via SerpAPI

SQLite-backed chat history

PDF invoice text extraction

```

```text
contract integration
What Contracts Typically Contain in AP Context:
Payment Terms
e.g., Net 30, early payment discounts, late fees.

Scope of Work / Deliverables
Useful to cross-check what’s being billed in the invoice.

Authorized Vendors or Contacts
Cross-verification with invoice vendor.

Termination Clauses
May affect if payment is due or the contract has ended.

✅ How Contract Data Can Be Used in Your Pipeline:
You can extract fields from the contract (as you've started to do) and:

✅ Validate invoice details (terms, vendor, scope) against the contract.

✅ Compare PO line items against contract deliverables.

✅ Use business rules to resolve conflicts (e.g., contract overrides PO terms).

✅ Feed extracted data into your run_llm_invoice_agent function as part of combined_data.

💡 Suggested Integration:
Treat contract as an optional fourth input in your UI and backend.

Extend combined_data to include "contract_details" alongside invoice, PO, and rules.

Enhance your tools (e.g., approval_process, verification) to optionally consult contract data.

```
Fine-tuning
``` text
Flow:
Upload invoice (PDF or image)

Extract text → map fields via TinyLlama

Run task: Verification / PO Match / Approval / Payment

Let me know if you want:

🧠 The mapping model trained on mapping_finetune_data.jsonl

📊 Confidence scores or visual indicators in the UI

🧊 GGUF export of mapping model for local/offline inference

You're now running a structured, production-grade invoice agent backed by your fine-tuned TinyLlama 🧾

streamlit run ./ui/fine_tune_hybrid.py
streamlit run ./ui/streamlit_app.py 

https://colab.research.google.com/drive/1NK6eWUVDHt3lCOaQE1aKIzKeR46_IyZX#scrollTo=iq8KMTnEXqnZ 
used for slm fine-tuning google colab

You're now running:

A fine-tuned TinyLlama model

With structured mapping

Driving a full invoice processing toolchain

All running securely and locally or on Colab
✅ RAG on Top of Fine-Tuned SLM
Yes, you can layer RAG over your fine-tuned TinyLlama by:

🧠 Creating a vector store (e.g., FAISS or ChromaDB)

📄 Indexing knowledge (PO policies, vendor rules, etc.)

🔄 At runtime: search → retrieve → inject into prompt context

Since you're using a small model, you may want to:

Limit context size

Use fuzzy matching or short chunks (to avoid context overflow)

Would you prefer:

A minimal working RAG example integrated into your toolchain?

Or a standalone RAG module with LangChain or plain Python?

Let me know your preferred direction and I’ll start with batch processing or RAG first.
```