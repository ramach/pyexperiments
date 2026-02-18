## Target architecture (wallet-preliminary)

``` txt
Core components

1) Channels

Mobile/Web app (consumer)

Ops portal (support, disputes, limits, KYC)

2) Edge

API Gateway (Kong/APISIX) + WAF rules

Keycloak (OIDC) for user auth + MFA for ops

3) Wallet Core (your domain)

Wallet Service (accounts, currencies, balances view)

Ledger Service (double-entry accounting; can be Apache Fineract here)

Transaction Service (wallet transaction history, search, filters)

Payout Service (withdrawals to linked bank accounts)

Top-up Service (bank transfer + card top-ups orchestration)

Statement Service (PDF/CSV generation + storage)

4) Payments / Rail integration
Payment Hub EE as the switch/connector layer

Bank transfer rails (ACH/SEPA/RTGS or local switch)

Payout rails for withdrawals

Card processor/PSP for debit/credit top-ups (Stripe/Adyen/etc.)

Important: keep cards out of your environment to avoid heavy PCI scope (use hosted components + tokenization)

5) Data + eventing

PostgreSQL (transactional)

Redis (real-time balance cache + pagination tokens)

Kafka/RabbitMQ (events, async updates, statement jobs)

Object store (MinIO/S3) for statements + KYC docs

6) Observability + audit

OpenTelemetry + Prometheus/Grafana

OpenSearch/ELK for logs & audit search

Append-only wallet_event stream for timelines

```
## Multi-currency wallet model (clean and scalable)

```txt
Entities

Customer

Wallet (one per customer)

Wallet Account (one per currency per wallet)
Example: Wallet 123 has USD, EUR, KES sub-accounts.

Bank Accounts (linked payout instruments)

Payment Instruments (card tokens stored at PSP, not you)

Balance strategy (real-time display)

Show balances as:

Available (spendable)

Reserved/Held (pending withdrawals, chargebacks, etc.)

Ledger/Posted (accounting truth)
```
## Rule of thumb

### Ledger (Fineract) is source of truth.

1. For “real-time display”, compute Available as:
2. posted_balance - reserved_balance
3. and cache it in Redis, updated by events.

### Transaction history + filtering/search

1. Ledger entries (accounting)
2. User-friendly transactions (what the user expects to see)

### Recommended: maintain a wallet_transaction table that is derived from ledger events

``` python

transaction_id, type (TOPUP_CARD, TOPUP_BANK, WITHDRAWAL, FX_CONVERSION, FEE)

currency, amount, status

counterparty, rail_reference, created_at

searchable_text (merchant/beneficiary/reference)

Index for:

customer_id + created_at

status

type

full-text search on searchable_text (Postgres FTS)

```
## Top-up methods (bank transfer + cards)

#### A) Bank transfer top-up (push payments)

##### Flow

1. User requests “Add money via bank transfer” → app shows:

   virtual IBAN/account/reference (best)
   or instruction with unique reference

2. Incoming bank credit hits your bank/switch → Payment Hub EE receives notification/statement feed

3. Payment Hub EE → Wallet Top-up Service: “Inbound credit received”

#### Top-up Service:

     1. validates reference, AML/limits, duplicate detection

     2. posts ledger credit to user wallet account

     3. Wallet balance updates in real-time via event

``` txt
B) Debit/Credit card top-up

Best practice

Use PSP-hosted payment page / mobile SDK tokenization.

You receive payment intent result + token, never raw PAN.

Flow

App → Top-up Service: create card top-up intent

Top-up Service → PSP: create PaymentIntent/Session

App completes payment at PSP

PSP webhook → your Webhook endpoint (signed)

Top-up Service verifies webhook → posts ledger credit to wallet
(optionally after risk checks)

Chargebacks

Treat chargebacks as a separate negative event:

move funds from user wallet to a “chargeback/negative balance” path

or reserve a rolling amount for card-funded balances (risk policy)

```

### Withdrawal to linked bank account

#### Flow

1. User links bank account (verify ownership where possible)

2. User requests withdrawal (amount, currency)

3. Payout Service:

     limits/risk checks

     reserve funds (hold)

     submit payout to Payment Hub EE

4. Payment Hub EE executes payout on bank rail (async status)

5. On success: finalize ledger (posted debit)

6. On fail: release reserve

#### Real-world necessity
1. Withdrawals are asynchronous. Always show status: Pending / Processing / Completed / Failed.

### Ledger postings (simple, robust pattern)

``` txt

Assume each wallet currency is a liability account. Use a clearing account for in-flight.

1) Card top-up success (credit wallet)

DR Card Settlement Clearing (Asset)

CR Customer Wallet Liability (e.g., USD wallet)

Fees (if you charge user):

DR Customer Wallet Liability (fee)

CR Fee Income

2) Bank transfer top-up (inbound credit)

DR Bank Account (Asset) / Bank Clearing

CR Customer Wallet Liability

3) Withdrawal initiation (reserve/hold)
Reserve without final debit (two options):

Option A (recommended): internal reserve table + no ledger posting yet

Keep ledger clean; show reserved in wallet service.

Post to ledger only on completion.

Option B: ledger hold via clearing

DR Customer Wallet Liability

CR Withdrawal Clearing Liability
Then on completion:

DR Withdrawal Clearing Liability

CR Bank Settlement Account / Payables

If failed:

Reverse the hold posting.

4) Withdrawal completion

DR Customer Wallet Liability (or clearing if you used it)

CR Bank Settlement Payable / Bank Account
```

### Statement generation (PDF/CSV)
```txt
CSV export (fast, cheap)

Query wallet_transaction by date range + filters

Produce CSV

Store in MinIO with short-lived signed URL

PDF statement (more formal)

Statement Service pulls:

opening balance, closing balance

transaction list

fees summary

Render via a server-side generator (e.g., ReportLab or WeasyPrint)

Store PDF in MinIO; return link

Important

Generate statements asynchronously (job queue) so you don’t block the API.

Keep the exact statement data snapshot used for the PDF (for audit).
```
### Security & compliance notes

```nginx
Cards (debit/credit top-up)

If you accept cards directly, PCI scope explodes.

To stay lean:

use PSP-hosted fields/SDK

store only PSP tokens + last4 + brand

lock down webhook verification + replay protection

Money safety controls you want early

Idempotency keys on:

create top-up intent

process PSP webhook

submit withdrawal

process rail callbacks

Immutable audit log for:

bank account changes

limits changes

refunds/chargebacks

Maker-checker for:

manual balance adjustments

refunds

limit overrides

```
#### MVP Order
```nginx
Single currency wallet (USD only), real-time balances, history

Bank transfer top-ups (lowest compliance complexity)

Withdrawals to linked bank accounts (async)

Multi-currency (separate sub-accounts per currency)

Card top-ups (PSP + webhooks + chargeback handling)

Statements (CSV first, then PDF)
```

### Service map and responsibilities

``` txt
wallet-api (BFF / public API)

Auth (Keycloak OIDC), user-facing endpoints

Fan-out to internal services

Enforces idempotency keys at the edge

B. wallet-core

Wallet + multi-currency sub-accounts (USD/EUR/…)

Computes balances: available / reserved / posted

Owns holds/reservations (recommended)

C. ledger-adapter

Only component allowed to talk to Fineract

Posts journal entries, fetches posted balances

Emits ledger.posted events

topup-service

Bank transfer top-ups (inbound credits)

Card top-ups (PSP intents + webhook verification)

Normalizes top-up status and triggers ledger postings

E. payout-service

Withdrawals to linked bank accounts

Submits payout via Payment Hub EE

Processes async callbacks; finalizes ledger

F. tx-history-service

Maintains user-friendly wallet_transaction read model

Supports filtering, search, pagination

G. statement-service

Generates CSV/PDF asynchronously (jobs)

Stores to MinIO; returns signed download links

H. ops-portal (separate UI)

manual review queue, refunds, account changes (maker-checker)

audit timeline

```

#### Public API sketch (high-signal endpoints)

#### Wallets & balances

##### GET /v1/wallets/me

```json
{
  "wallet_id": "w_123",
  "currencies": ["USD","EUR"],
  "default_currency": "USD"
}
```

##### GET /v1/wallets/me/balances

```json
{
  "balances": [
    {"currency":"USD","posted":"1200.50","reserved":"100.00","available":"1100.50","as_of":"2026-01-23T10:10:00Z"},
    {"currency":"EUR","posted":"10.00","reserved":"0.00","available":"10.00","as_of":"2026-01-23T10:10:00Z"}
  ]
}
```

##### Transaction history
``` txt
GET /v1/wallets/me/transactions?currency=USD&type=TOPUP_CARD&status=COMPLETED&from=2026-01-01&to=2026-01-23&q=uber&limit=50&cursor=...
```
``` json
{
  "items": [
    {
      "tx_id":"tx_abc",
      "type":"TOPUP_CARD",
      "status":"COMPLETED",
      "currency":"USD",
      "amount":"50.00",
      "fee":"1.50",
      "net_amount":"48.50",
      "reference":"psp_pi_123",
      "counterparty":"VISA **** 4242",
      "created_at":"2026-01-23T09:00:00Z"
    }
  ],
  "next_cursor":"..."
}
```
##### Bank accounts (withdrawal destination)
```nginx
POST /v1/wallets/me/bank-accounts
```
``` json
{"bank_account_id":"ba_123","status":"PENDING_VERIFICATION"}
```

#### Internal event model (Kafka/RabbitMQ topics)
``` txt
Use an outbox table (see schema) and publish events reliably.

Recommended topics

wallet.events (domain events)

ledger.events (posted confirmations)

payments.callbacks (Payment Hub EE callbacks normalized)

psp.webhooks (raw PSP events, then normalized)

Key events (examples)

wallet.hold.created (reserve funds)

wallet.hold.released

topup.created

topup.completed

withdrawal.submitted

withdrawal.completed

ledger.posted (journal entry posted in Fineract)
```
#### Database schema draft (Postgres)

``` txt
This assumes wallet-core owns holds and tx-history is a derived read model.

Core: wallets & accounts

```
``` sql
CREATE TABLE wallet (
  wallet_id        TEXT PRIMARY KEY,
  customer_id      TEXT NOT NULL UNIQUE,
  default_currency TEXT NOT NULL,
  status           TEXT NOT NULL CHECK (status IN ('ACTIVE','SUSPENDED','CLOSED')),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE wallet_account (
  wallet_account_id TEXT PRIMARY KEY,
  wallet_id         TEXT NOT NULL REFERENCES wallet(wallet_id),
  currency          TEXT NOT NULL,
  fineract_account_id TEXT, -- link to Fineract account for this currency
  status            TEXT NOT NULL CHECK (status IN ('ACTIVE','SUSPENDED')),
  UNIQUE(wallet_id, currency)
);
Holds / reservations (recommended)
sql
Copy code
CREATE TABLE wallet_hold (
  hold_id      TEXT PRIMARY KEY,
  wallet_id    TEXT NOT NULL REFERENCES wallet(wallet_id),
  currency     TEXT NOT NULL,
  amount       NUMERIC(18,2) NOT NULL CHECK (amount > 0),
  reason       TEXT NOT NULL, -- WITHDRAWAL_PENDING, CHARGEBACK_RISK, etc.
  status       TEXT NOT NULL CHECK (status IN ('ACTIVE','RELEASED','CONSUMED')),
  idempotency_key TEXT,
  related_id   TEXT, -- withdrawal_id/topup_id/etc
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  released_at  TIMESTAMPTZ
);

CREATE INDEX idx_hold_wallet_currency_status ON wallet_hold(wallet_id, currency, status);
Money-moving objects
CREATE TABLE topup (
  topup_id     TEXT PRIMARY KEY,
  wallet_id    TEXT NOT NULL REFERENCES wallet(wallet_id),
  method       TEXT NOT NULL CHECK (method IN ('BANK_TRANSFER','CARD')),
  currency     TEXT NOT NULL,
  amount       NUMERIC(18,2) NOT NULL CHECK (amount > 0),
  fee          NUMERIC(18,2) NOT NULL DEFAULT 0,
  status       TEXT NOT NULL CHECK (status IN ('CREATED','PENDING','COMPLETED','FAILED','CANCELLED')),
  reference    TEXT, -- PSP payment intent id or bank reference
  idempotency_key TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_topup_idem ON topup(wallet_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE withdrawal (
  withdrawal_id TEXT PRIMARY KEY,
  wallet_id     TEXT NOT NULL REFERENCES wallet(wallet_id),
  bank_account_id TEXT NOT NULL,
  currency      TEXT NOT NULL,
  amount        NUMERIC(18,2) NOT NULL CHECK (amount > 0),
  fee           NUMERIC(18,2) NOT NULL DEFAULT 0,
  status        TEXT NOT NULL CHECK (status IN ('CREATED','PENDING','SUBMITTED','IN_PROGRESS','COMPLETED','FAILED','CANCELLED','REVERSED','MANUAL_REVIEW')),
  hub_reference TEXT, -- Payment Hub EE reference
  hold_id       TEXT, -- link to wallet_hold
  idempotency_key TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_withdrawal_idem ON withdrawal(wallet_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
Transaction history (read model)
sql
Copy code
CREATE TABLE wallet_transaction (
  tx_id        TEXT PRIMARY KEY,
  wallet_id    TEXT NOT NULL REFERENCES wallet(wallet_id),
  type         TEXT NOT NULL,  -- TOPUP_CARD, TOPUP_BANK, WITHDRAWAL, FEE, FX, ADJUSTMENT
  status       TEXT NOT NULL,  -- PENDING, COMPLETED, FAILED, REVERSED
  currency     TEXT NOT NULL,
  amount       NUMERIC(18,2) NOT NULL,
  fee          NUMERIC(18,2) NOT NULL DEFAULT 0,
  net_amount   NUMERIC(18,2) NOT NULL,
  reference    TEXT,
  counterparty TEXT,
  searchable_text TEXT,
  occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tx_wallet_time ON wallet_transaction(wallet_id, occurred_at DESC);
CREATE INDEX idx_tx_filters ON wallet_transaction(wallet_id, currency, type, status, occurred_at DESC);
-- optional: full text search
-- CREATE INDEX idx_tx_fts ON wallet_transaction USING GIN (to_tsvector('simple', coalesce(searchable_text,'')));
Outbox (reliability)
sql
Copy code
CREATE TABLE outbox_event (
  event_id     TEXT PRIMARY KEY,
  aggregate_id TEXT NOT NULL,
  aggregate_type TEXT NOT NULL,
  event_type   TEXT NOT NULL,
  payload      JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  published_at TIMESTAMPTZ
);

CREATE INDEX idx_outbox_unpublished ON outbox_event(published_at) WHERE published_at IS NULL;
5
```
#### State machines + safe retries
A) Card top-up state machine
``` txt
States: CREATED → PENDING → COMPLETED | FAILED | CANCELLEDFlow

CREATED: intent created (idempotency enforced)

PENDING: PSP payment initiated

PSP webhook arrives:

verify signature + timestamp + nonce

dedupe (store webhook event id)

On payment_succeeded:

topup.status = COMPLETED

trigger ledger posting (credit wallet)

On payment_failed/canceled:

mark FAILED or CANCELLED

Retry rules

Webhook processing: retryable if internal error, but dedupe by PSP event id

Ledger posting: retryable with idempotency key ledger:topup:{topup_id}

B) Bank transfer top-up state machine
States: CREATED(optional instructions) → PENDING_MATCH → COMPLETED | MANUAL_REVIEW | FAILED

Flow

Inbound credit notification from Payment Hub EE (or statement ingest)

Match by reference / virtual account

If matched:

COMPLETED + ledger posting credit

If not matched:

MANUAL_REVIEW case (ops can link to wallet)

Retry

Statement ingestion: safe to re-run; dedupe by (rail_ref, amount, currency, value_date)

C) Withdrawal state machine
States:
CREATED → PENDING → SUBMITTED → IN_PROGRESS → COMPLETED
or terminal: FAILED | REVERSED | CANCELLED | MANUAL_REVIEW

Flow

CREATED: request validated

PENDING: risk/limits checks passed

Create hold: wallet_hold ACTIVE

SUBMITTED: sent to Payment Hub EE (idempotency key hub:withdrawal:{id})

IN_PROGRESS: rail accepted

COMPLETED: finalize

consume hold

ledger posting debit + fees

FAILED: release hold

Retry rules

Submitting to Hub: retry on timeout/5xx, but must use idempotency key

Callback handling: dedupe by hub event id; ignore older status transitions

6) Ledger integration (Fineract) – minimal pattern
Recommended approach (cleanest)
Holds/reservations live in your DB (wallet_hold)

Fineract only sees posted events:

top-up completion

withdrawal completion

fees (at completion)

reversals/refunds

This avoids cluttering ledger with temporary holds and makes audits easier.

Posting examples (conceptual)
Top-up completed (USD 50, fee 1.50)

Credit customer wallet liability +48.50 (or full 50 with separate fee debit—pick one convention)

Credit fee income +1.50

Debit settlement/clearing +50.00

Withdrawal completed (USD 100, fee 2.00)

Debit customer wallet liability 102.00

Credit bank settlement payable 100.00

Credit fee income 2.00

(Exact Fineract account mapping depends on how you configure COA/products.)

7) Deployment topology (Kubernetes-ready)
Namespaces
edge : gateway, WAF ingress

identity : Keycloak

core : wallet-core, topup, payout, ledger-adapter, tx-history, statement

data : Postgres, Redis, Kafka/RabbitMQ (or managed externally)

observability : Prometheus, Grafana, OpenTelemetry collector, OpenSearch

Network policy (high level)
edge → core (only via gateway)

core services talk to each other via internal DNS

Only ledger-adapter can reach Fineract

Only topup-service can reach PSP (outbound)

Only payout-service can reach Payment Hub EE (or via internal connector)

MinIO accessible only from statement-service + KYC service

Secrets
Use Vault or sealed-secrets

Separate secrets per service:

PSP webhook secret

Hub callback secret / mTLS certs

DB creds (short-lived if possible)

Observability
Every service emits:

structured logs (JSON)

traces (OpenTelemetry)

metrics (Prometheus)

Correlation IDs:

x-request-id from gateway

propagate into events and ledger postings

8) “Next implementation step” checklist (actionable)
Stand up: Keycloak + Gateway + Postgres + Redis + Kafka/RMQ + MinIO

Implement wallet-core:

wallets + wallet_account per currency

holds

balances endpoint (posted from ledger + reserved from holds)

Implement ledger-adapter:

Fineract account setup per currency

post journal entries with idempotency

Implement topup-service:

card intent creation (PSP)

webhook verification + dedupe

bank transfer matching ingest (via hub)

Implement payout-service:

create withdrawal + hold

submit to Payment Hub EE

callback handler + finalize/release hold

Implement tx-history-service read model:

consume events + build wallet_transaction

Implement statement-service:

async job + CSV first, then PDF

If you want, I can follow up with a full endpoint list (including internal admin endpoints), plus sample JSON schemas for each event and a sequence diagram for:

Card top-up (intent → PSP → webhook → ledger → balance update)

Withdrawal (hold → hub submit → callback → ledger finalize → statement)


```

```nginx
it in pieces and sanity-check it against:

your regulatory constraints,

your target countries/rails,

```
Here’s an absolute MVP we can build in small, safe increments, plus a concrete repo structure and “build order” that keeps you from over-engineering early.

#### Absolute MVP definition (smallest useful wallet)
##### Scope (MVP-0)
```nginx
Single currency wallet (start with USD or local currency)

Real-time balance (posted + reserved)

Transaction history (basic list, no full-text search yet)

Top-up via bank transfer only (inbound credit matching by unique reference)

Withdrawal to linked bank account (async, with hold/reserve)

Statement CSV export (PDF later)

Explicitly not in MVP-0

Cards (debit/credit), FX, multi-currency, chargebacks, complex AML

Fancy ops portal (just minimal admin endpoints/logs)

That’s enough to prove: ledger correctness, idempotency, reconciliation hooks.
```

##### Build order (incremental “slices”)

Slice 1 — Wallet core + balances (no money movement yet)
Goal: Create wallet, show balance, show empty history.

Wallet + wallet_account tables

Holds/reservations table

Balance endpoint: posted - reserved

Seed posted balance = 0 initially (ledger integration stub)

Slice 2 — Ledger adapter (Fineract) “posted truth”
Goal: When a transaction is marked posted, it becomes real money.

Implement ledger-adapter service that:

creates/links the Fineract account for wallet currency

posts journal entries idempotently

can query posted balance

Wallet balance reads posted from ledger-adapter + reserved from holds

Slice 3 — Bank transfer top-up (inbound only)
Goal: Credit wallet when an inbound bank transfer arrives.

Create “bank top-up instructions” endpoint returning a unique reference

Create an inbound credits ingest endpoint (simulates Payment Hub EE for now)

When inbound credit arrives:

validate reference → identify wallet

post ledger credit

append transaction history record

Slice 4 — Withdrawals to bank account (async)
Goal: Reserve funds, submit payout, complete/fail via callback.

Link bank account endpoint

Create withdrawal:

create hold

submit to “rail” (mock Payment Hub EE)

Callback endpoint updates status:

success → consume hold + post ledger debit + record tx

failure → release hold + record failure tx

Slice 5 — CSV Statements
Goal: Export filtered tx history to CSV, store, download.

Statement job → generate CSV from wallet_transaction

Store in MinIO/local + return signed/temporary URL

When that works end-to-end, you add:

Cards (Slice 6)

Multi-currency + FX (Slice 7)

PDF statements (Slice 8)

Real reconciliation (Slice 9)

### Repo structure (practical, not overkill)

Option A (recommended for gradual build): Modular monolith first
You still keep clean boundaries, but deploy one service initially.

```pgsql

remit-wallet/
  README.md
  docker-compose.yml
  .env.example

  apps/
    api/                      # FastAPI entrypoint (public API + internal admin)
      main.py
      deps.py
      routes/
        wallet.py
        topups.py
        withdrawals.py
        statements.py
        admin.py
      middleware/
        idempotency.py
        request_id.py

  core/
    config.py
    db.py
    models/                   # SQLAlchemy models
      wallet.py
      hold.py
      topup.py
      withdrawal.py
      transaction.py
      outbox.py
    services/                 # Business logic (domain)
      wallet_service.py
      ledger_service.py       # adapter interface (real or stub)
      topup_service.py
      payout_service.py
      statement_service.py
    integrations/
      fineract_client.py       # later real
      paymenthub_client.py     # later real
      storage.py              # MinIO/local abstraction
    events/
      publisher.py            # outbox publisher (later)
      schemas.py

  migrations/                 # Alembic
  tests/
    unit/
    integration/
  scripts/
    seed_dev.py
```
Why this is good: you can ship Slice 1–5 quickly, and later split into microservices if needed.

##### Option B: Microservices from day 1 (only if you must)
wallet-core, ledger-adapter, topup-service, payout-service, etc.
It’s doable, but you’ll spend more time on plumbing.

##### MVP-0 data model (minimal tables)
``` txt
wallet(wallet_id, customer_id, default_currency, status, created_at)

wallet_account(wallet_account_id, wallet_id, currency, ledger_account_ref, status)

wallet_hold(hold_id, wallet_id, currency, amount, status, reason, related_id, created_at)

wallet_transaction(tx_id, wallet_id, type, status, currency, amount, fee, net_amount, reference, occurred_at)

bank_account(bank_account_id, wallet_id, masked_details, status, created_at)

outbox_event(...) (optional in MVP; add when you introduce async processing)

Minimal API surface (MVP endpoints)
Wallet

GET /v1/wallets/me

GET /v1/wallets/me/balances

GET /v1/wallets/me/transactions?from&to&type&status&limit&cursor

Top-up (bank transfer)

GET /v1/topups/bank/instructions

POST /internal/inbound-credits (mock Payment Hub EE → you replace later)

Withdrawals

POST /v1/wallets/me/bank-accounts

POST /v1/withdrawals (creates hold + submits)

POST /internal/payout-callback (mock rail callback)

Statements

POST /v1/statements (CSV only)

GET /v1/statements/{id}

The two “must-not-break” rules (build these in early)
Idempotency keys on:

inbound credits ingest

create withdrawal

payout callback handling
(so retries don’t double-credit or double-debit)

Holds first, then payout

never submit withdrawal unless hold exists

if payout fails → release hold

Concrete implementation plan you can follow immediately
Start with Option A repo (modular monolith) + docker-compose for Postgres/Redis/MinIO

Implement Slice 1 endpoints + DB migrations + unit tests

Add ledger_service.py with two implementations:

InMemoryLedger (dev)

FineractLedger (later)

Add Slice 3 inbound credits ingest endpoint to simulate Payment Hub EE

Add Slice 4 withdrawal async flow (submit + callback)

Add Slice 5 CSV statement job (sync first, then async queue later)

If you want the next deliverable right now
language/framework you prefer for the API (I assumed Python + FastAPI), and I’ll respond with:

a ready-to-run skeleton (docker-compose + FastAPI files + SQLAlchemy models + Alembic baseline),

plus sample curl requests for each slice.

```
#### Absolute MVP definition (smallest useful wallet)

##### Scope (MVP-0)

1. Single currency wallet (start with USD or local currency)

2. Real-time balance (posted + reserved)

3. Transaction history (basic list, no full-text search yet)

4. Top-up via bank transfer only (inbound credit matching by unique reference)

5. Withdrawal to linked bank account (async, with hold/reserve)

6. Statement CSV export (PDF later)

##### Explicitly not in MVP-0

1. Cards (debit/credit), FX, multi-currency, chargebacks, complex AML

2. Fancy ops portal (just minimal admin endpoints/logs)

That’s enough to prove: ledger correctness, idempotency, reconciliation hooks.

##### Build order (incremental “slices”)

###### Slice 1 — Wallet core + balances (no money movement yet)

Goal: Create wallet, show balance, show empty history.

1. Wallet + wallet_account tables

2. Holds/reservations table

3. Balance endpoint: posted - reserved

4. Seed posted balance = 0 initially (ledger integration stub)

##### Slice 2 — Ledger adapter (Fineract) “posted truth”

Goal: When a transaction is marked posted, it becomes real money.

Implement ledger-adapter service that:

1. creates/links the Fineract account for wallet currency

2. posts journal entries idempotently

3. can query posted balance

Wallet balance reads posted from ledger-adapter + reserved from holds

##### Slice 3 — Bank transfer top-up (inbound only)

Goal: Credit wallet when an inbound bank transfer arrives.

1. Create “bank top-up instructions” endpoint returning a unique reference

2. Create an inbound credits ingest endpoint (simulates Payment Hub EE for now)

3. When inbound credit arrives:

4. validate reference → identify wallet

5. post ledger credit

6. append transaction history record

##### Slice 4 — Withdrawals to bank account (async)

Goal: Reserve funds, submit payout, complete/fail via callback.

1. Link bank account endpoint

2. Create withdrawal:

3. create hold

4. submit to “rail” (mock Payment Hub EE)

5. Callback endpoint updates status:

success → consume hold + post ledger debit + record tx

failure → release hold + record failure tx

##### Slice 5 — CSV Statements

Goal: Export filtered tx history to CSV, store, download.
Statement job → generate CSV from wallet_transaction

1. Store in MinIO/local + return signed/temporary URL

When that works end-to-end, add:

Cards (Slice 6)

Multi-currency + FX (Slice 7)

PDF statements (Slice 8)

Real reconciliation (Slice 9)

```pgsql
remit-wallet/
  README.md
  docker-compose.yml
  .env.example

  apps/
    api/                      # FastAPI entrypoint (public API + internal admin)
      main.py
      deps.py
      routes/
        wallet.py
        topups.py
        withdrawals.py
        statements.py
        admin.py
      middleware/
        idempotency.py
        request_id.py

  core/
    config.py
    db.py
    models/                   # SQLAlchemy models
      wallet.py
      hold.py
      topup.py
      withdrawal.py
      transaction.py
      outbox.py
    services/                 # Business logic (domain)
      wallet_service.py
      ledger_service.py       # adapter interface (real or stub)
      topup_service.py
      payout_service.py
      statement_service.py
    integrations/
      fineract_client.py       # later real
      paymenthub_client.py     # later real
      storage.py              # MinIO/local abstraction
    events/
      publisher.py            # outbox publisher (later)
      schemas.py

  migrations/                 # Alembic
  tests/
    unit/
    integration/
  scripts/
    seed_dev.py

```
##### Option B: Microservices from day 1 (only if you must)

wallet-core, ledger-adapter, topup-service, payout-service, etc.
It’s doable, but you’ll spend more time on plumbing.

##### MVP-0 data model (minimal tables)

1. wallet(wallet_id, customer_id, default_currency, status, created_at)

2. wallet_account(wallet_account_id, wallet_id, currency, ledger_account_ref, status)

3. wallet_hold(hold_id, wallet_id, currency, amount, status, reason, related_id, created_at)

4. wallet_transaction(tx_id, wallet_id, type, status, currency, amount, fee, net_amount, reference, occurred_at)

5. bank_account(bank_account_id, wallet_id, masked_details, status, created_at)

outbox_event(...) (optional in MVP; add when you introduce async processing)

##### Minimal API surface (MVP endpoints)

1. Wallet

GET /v1/wallets/me

GET /v1/wallets/me/balances

GET /v1/wallets/me/transactions?from&to&type&status&limit&cursor

2. Top-up (bank transfer)

GET /v1/topups/bank/instructions

POST /internal/inbound-credits (mock Payment Hub EE → you replace later)

3. Withdrawals

POST /v1/wallets/me/bank-accounts

POST /v1/withdrawals (creates hold + submits)

POST /internal/payout-callback (mock rail callback)

4. Statements

POST /v1/statements (CSV only)

GET /v1/statements/{id}

#### Project layout
```txt
remit-wallet/
docker-compose.yml
.env.example
README.md

requirements.txt

app/
__init__.py
main.py
config.py
db.py

    models/
      __init__.py
      wallet.py
      hold.py
      transaction.py
      bank_account.py
      topup.py
      withdrawal.py

    schemas/
      __init__.py
      wallet.py
      topup.py
      withdrawal.py
      statement.py
      common.py

    services/
      __init__.py
      wallet_service.py
      topup_service.py
      payout_service.py
      statement_service.py
      idempotency_service.py

    api/
      __init__.py
      deps.py
      routes_wallet.py
      routes_topups.py
      routes_withdrawals.py
      routes_statements.py
      routes_internal.py

alembic.ini
alembic/
env.py
script.py.mako
versions/
0001_init.py
```

#### Use Fineract as “Ledger-of-Record” via Journal Entries

What you keep locally (your wallet DB)

1. Wallet, bank_account, withdrawal, wallet_transaction (for UI/history)

2. Idempotency keys + request logs

3. A lightweight “posting status” table to track what was posted to Fineract

What Fineract does

1. Chart of Accounts (COA)

2. Double-entry posting (debits/credits)

3. Trial balance / GL reporting, accounting closures, audit expectations

Fineract exposes accounting endpoints including journal entries (“Create Balanced Journal Entries”, “List Journal Entries”).

1. Mapping for bank transfer top-up (double-entry)

When a customer tops up USD 100:

1. Debit: Settlement / Bank clearing (asset) 100

2. Credit: Customer wallet liability (liability) 100

You create a balanced journal entry in Fineract with:

1. officeId (or default office)

2. transactionDate

3. credits + debits lines

4. a reference / “externalId” / “transactionIdentifier” for idempotency (depends on your Fineract distribution)

This is exactly the conceptual flow used in Fineract accounting/journal entries.

#### Ledger Service (Fineract)

A) Keep existing wallet tables as the channel layer

…and add a ledger_postings table:

1. posting_id

2. wallet_tx_id

3. ledger_provider = “fineract”

4. provider_ref

status = PENDING / POSTED / FAILED

5. last_error

6. timestamps

B) On /v1/topups/bank-transfer:

1. Write wallet_transaction locally (status = PENDING_LEDGER)

2. Post journal entry to Fineract (idempotent key = tx_id)

3. If success → mark local tx COMPLETED + ledger_postings POSTED

4. If failure → local tx stays PENDING_LEDGER and can be retried

This gives:

1. Strong audit trail

2. Retry safety

3. Clear separation between channel UX and accounting truth

##### Target end-state

``` txt
wallet DB remains the “channel layer”

user-visible transactions, workflow states, idempotency

retry queue / posting status

UI & APIs stay fast and simple

Fineract becomes the accounting truth

COA + GL reporting

“double entry” for every wallet movement

audit trail in Fineract’s journal entries

```

##### Minimal ledger_service module structure

``` python
api/app/ledger/
  __init__.py
  config.py
  fineract_client.py
  ledger_service.py
  models.py   # (optional) provider posting table models

```

##### Ledger Service Table

Add a “posting status” table in your wallet DB

Add ledger_posting table that tracks:

1. wallet_tx_id

2. provider (“fineract”)

3. provider_ref (journalEntryId / resourceId)

   status (PENDING/POSTED/FAILED)

4. last_error

This gives:

1. safe retries

2. visibility in UI

3. audit trail