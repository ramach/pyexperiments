```graphql
llm_ap_ar_agent/
├── agents/
│   ├── llm_ap_ar_agent.py            # Main agent orchestration
│   ├── openai_function_handler.py    # OpenAI function schemas & tool handlers
├── chains/
│   ├── ap_chain.py                   # Chain logic for Accounts Payable
│   ├── ar_chain.py                   # Chain logic for Accounts Receivable
├── retrievers/
│   ├── query_optimizer.py            # Token-aware document retrieval + scoring
├── vectorstore/
│   ├── faiss_store.py                # FAISS-based vectorstore setup
├── public_data/
│   ├── vendor_scraper.py             # (Optional) Simulated vendor data scraping
├── main.py                           # CLI entry point (with args)
├── tests/
│   ├── test_llm_ap_ar_agent.py       # Unit tests for agent logic
│   ├── test_query_optimizer.py       # Tests for token/document filtering
├── requirements.txt
├── README.md
```
```bash
pip install -r requirements.txt
```
requirements.txt (sample):

```nginx
langchain
openai
faiss-cpu
tiktoken
unstructured
beautifulsoup4
chromadb  # Optional if switching to Chroma
```

How to Run

```bash
python main.py \
  --query "List all invoices due in April" \
  --mode ap \
  --vectorstore faiss
```

```
Modes supported:

ap – Accounts Payable

ar – Accounts Receivable

vendor – Vendor-specific

public – Scraped or public financial data
```

```txt
Key Features

✅ OpenAI function calling
✅ Context optimization with token limits
✅ LangChain-based tool agent
✅ Vectorstore retrieval (FAISS or Chroma)
✅ Testable with unittest and mocks
```

```bash
python -m unittest discover tests
```

```txt
requirements.txt

pip install -r requirements.txt --force-reinstall

# Core LLM Agent Stack
langchain==0.0.331
openai<1.0.0           # Required for compatibility with older LangChain
tiktoken

# Embeddings + Vector DB
faiss-cpu              # Or faiss-gpu if you prefer
chromadb

# Public Data and Utilities
google-search-results  # For SerpAPI (GoogleSearch)
python-dotenv
requests
beautifulsoup4

# Web UI
streamlit
pydantic<2.0           # Avoid Pydantic v2 unless LangChain fully supports it

# Optional for extended features or future LLM routers
sentence-transformers

```

```mermaid
graph TD
    Start(Start AP Process)

    subgraph Stage 1: Invoice Receipt & Capture
        S1_Receive[Receive Vendor Invoice ⚙️<br/>Hook: PreCapture Validation]
        S1_Digitize[Digitize & Extract Data PO# ⚙️<br/>Hooks: PostCapture Enrich,<br/>Duplicate Check]
    end

    subgraph Stage 2: Validation & Matching
        S2_Retrieve[Retrieve PO & GRN/Svc Accpt ⚙️<br/>Hook: PreMatch Validation]
        S2_Match{Perform 3-Way Match?<br/> ⚙️ Hook: Match Tolerance Rules}
    end

    subgraph Stage 3: Exception Handling
        S3_Identify[Identify Match Discrepancy]
        S3_Route[Route for Resolution ⚙️<br/>Hooks: Exception Routing,<br/>Escalation]
        S3_Document[Document Resolution]
    end

    subgraph Stage 4: Approval
        S4_RouteApproval[Route for Formal Approval ⚙️<br/>Hooks: Approval Matrix,<br/>PreApproval Budget Check]
        S4_Decision{Invoice Approved?}
        S4_Reject[Address Rejection Reason]
    end

    subgraph Stage 5: Coding & Posting
        S5_Code[Assign/Confirm GL Codes ⚙️<br/>Hook: GLCoding Validation]
        S5_Post[Post Invoice to Ledgers ⚙️<br/>Hook: PrePosting Compliance Check]
    end

    subgraph Stage 6: Payment Processing
        S6_Select[Select Invoices for Payment ⚙️<br/>Hook: Payment Selection Rules]
        S6_Authorize[Authorize Payment Batch ⚙️<br/>Hook: Payment Batch Approval]
        S6_Execute[Execute Payment ⚙️<br/>Hook: PrePayment Fraud Check]
        S6_Record[Record Payment & Send Remittance ⚙️<br/>Hook: PostPayment Notification]
    end

    subgraph Stage 7: Archiving
        S7_Archive[File/Archive Documents ⚙️<br/>Hook: Archive Metadata Tagging]
    end

    End(End AP Process)

    %% Workflow Connections
    Start --> S1_Receive
    S1_Receive --> S1_Digitize
    S1_Digitize --> S2_Retrieve
    S2_Retrieve --> S2_Match

    S2_Match -- No --> S3_Identify
    S3_Identify --> S3_Route
    S3_Route --> S3_Document
    %% Loop back for re-validation
    S3_Document --> S2_Retrieve

    S2_Match -- Yes --> S4_RouteApproval
    S4_RouteApproval --> S4_Decision

    S4_Decision -- No --> S4_Reject
    %% Loop back for rework/exception
    S4_Reject --> S3_Identify 

    S4_Decision -- Yes ⚙️<br/>Hook: PostApproval Notification --> S5_Code
    S5_Code --> S5_Post
    S5_Post --> S6_Select
    S6_Select --> S6_Authorize
    S6_Authorize --> S6_Execute
    S6_Execute --> S6_Record
    S6_Record --> S7_Archive
    S7_Archive --> End
```
