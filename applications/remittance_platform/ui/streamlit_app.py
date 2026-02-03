import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "dev-internal-key")

st.set_page_config(page_title="Wallet MVP UI", layout="wide")
st.title("Digital Wallet MVP")

def api_get(path, **kwargs):
    return requests.get(f"{API_BASE}{path}", timeout=15, **kwargs)

def api_post(path, json=None, headers=None):
    return requests.post(f"{API_BASE}{path}", json=json, timeout=15, headers=headers or {})

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Create Wallet")
    customer_id = st.text_input("customer_id", value="cust_001")
    currency = st.selectbox("default_currency", ["USD","EUR","KES"], index=0)
    if st.button("Create wallet"):
        r = api_post("/v1/wallets", json={"customer_id": customer_id, "default_currency": currency})
        st.write(r.status_code, r.json() if r.ok else r.text)

with col2:
    st.subheader("2) Load Wallet")
    wallet_id = st.text_input("wallet_id", value="")
    if st.button("Get balances"):
        r = api_get(f"/v1/wallets/{wallet_id}/balances")
        st.write(r.status_code, r.json() if r.ok else r.text)

st.divider()

c1, c2, c3 = st.columns([1,1,1])

with c1:
    st.subheader("3) Bank Top-up Instructions")
    topup_currency = st.selectbox("Topup currency", ["USD","EUR","KES"], index=0)
    if st.button("Show instructions"):
        r = api_get("/v1/topups/bank/instructions", params={"wallet_id": wallet_id, "currency": topup_currency})
        st.write(r.status_code, r.json() if r.ok else r.text)

with c2:
    st.subheader("4) Ingest inbound credit (internal mock)")
    rail_tx_id = st.text_input("rail_tx_id", value="rail_001")
    amount = st.text_input("amount", value="50.00")
    in_currency = st.selectbox("Inbound currency", ["USD","EUR","KES"], index=0)
    if st.button("Ingest inbound credit"):
        headers = {"X-Internal-API-Key": INTERNAL_KEY}
        payload = {"reference": f"WLT-{wallet_id}", "wallet_id": wallet_id, "currency": in_currency, "amount": amount, "rail_tx_id": rail_tx_id}
        r = api_post("/internal/inbound-credits", json=payload, headers=headers)
        st.write(r.status_code, r.json() if r.ok else r.text)

with c3:
    st.subheader("5) Transactions")
    if st.button("Refresh tx list"):
        r = api_get(f"/v1/wallets/{wallet_id}/transactions", params={"limit": 200})
        if r.ok:
            df = pd.DataFrame(r.json()["items"])
            st.dataframe(df, use_container_width=True)
        else:
            st.write(r.status_code, r.text)

st.divider()

c4, c5 = st.columns([1,1])

with c4:
    st.subheader("6) Link bank account")
    country = st.text_input("country", value="US")
    ba_currency = st.selectbox("bank account currency", ["USD","EUR","KES"], index=0, key="ba_curr")
    acct = st.text_input("account_number (MVP stores masked)", value="1234567890")
    if st.button("Add bank account"):
        r = api_post(f"/v1/wallets/{wallet_id}/bank-accounts", json={"country": country, "currency": ba_currency, "account_number": acct})
        st.write(r.status_code, r.json() if r.ok else r.text)

    if st.button("List bank accounts"):
        r = api_get(f"/v1/wallets/{wallet_id}/bank-accounts")
        st.write(r.status_code, r.json() if r.ok else r.text)

with c5:
    st.subheader("7) Create withdrawal + callback (internal mock)")
    bank_account_id = st.text_input("bank_account_id", value="")
    wd_currency = st.selectbox("withdrawal currency", ["USD","EUR","KES"], index=0, key="wd_curr")
    wd_amount = st.text_input("withdraw amount", value="10.00")
    idem = st.text_input("Idempotency-Key (optional)", value="wd_01")

    if st.button("Create withdrawal"):
        headers = {"Idempotency-Key": idem} if idem else {}
        payload = {"wallet_id": wallet_id, "bank_account_id": bank_account_id, "currency": wd_currency, "amount": wd_amount}
        r = requests.post(f"{API_BASE}/v1/withdrawals", json=payload, headers=headers, timeout=15)
        st.write(r.status_code, r.json() if r.ok else r.text)

    withdrawal_id = st.text_input("withdrawal_id for callback", value="")
    cb_status = st.selectbox("callback status", ["COMPLETED","FAILED"])
    cb_rail = st.text_input("callback rail_tx_id", value="rail_payout_001")

    if st.button("Send payout callback (internal)"):
        headers = {"X-Internal-API-Key": INTERNAL_KEY}
        payload = {"withdrawal_id": withdrawal_id, "status": cb_status, "rail_tx_id": cb_rail}
        r = api_post("/internal/payout-callback", json=payload, headers=headers)
        st.write(r.status_code, r.json() if r.ok else r.text)

st.divider()

st.subheader("8) Download statement CSV")
if st.button("Get statement.csv link"):
    st.write(f"{API_BASE}/v1/wallets/{wallet_id}/statement.csv")
    st.info("Open the link in a browser to download. You can also curl it.")
