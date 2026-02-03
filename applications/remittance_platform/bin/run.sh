curl -s -X POST http://localhost:8000/v1/wallets \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"cust_001","default_currency":"USD"}' | jq

export WALLET_ID=wlt_xxxxxxxxxxxxxxxx

curl -s http://localhost:8000/v1/wallets/$WALLET_ID/balances | jq
curl -s "http://localhost:8000/v1/wallets/$WALLET_ID/transactions?limit=50" | jq

curl -s "http://localhost:8000/v1/topups/bank/instructions?wallet_id=$WALLET_ID&currency=USD" | jq

curl -s -X POST http://localhost:8000/internal/inbound-credits \
  -H "Content-Type: application/json" \
  -H "X-Internal-API-Key: dev-internal-key" \
  -d '{"reference":"WLT-'$WALLET_ID'","wallet_id":"'$WALLET_ID'","currency":"USD","amount":"50.00","rail_tx_id":"rail_001"}' | jq

curl -s -X POST http://localhost:8000/v1/wallets/$WALLET_ID/bank-accounts \
  -H "Content-Type: application/json" \
  -d '{"country":"US","currency":"USD","account_number":"1234567890"}' | jq

curl -s http://localhost:8000/v1/wallets/$WALLET_ID/bank-accounts | jq

export BA_ID=ba_xxxxxxxxxxxxxxxx

curl -s -X POST http://localhost:8000/v1/withdrawals \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: wd_01" \
  -d '{"wallet_id":"'$WALLET_ID'","bank_account_id":"'$BA_ID'","currency":"USD","amount":"10.00"}' | jq

export WD_ID=wd_xxxxxxxxxxxxxxxx

curl -s -X POST http://localhost:8000/internal/payout-callback \
  -H "Content-Type: application/json" \
  -H "X-Internal-API-Key: dev-internal-key" \
  -d '{"withdrawal_id":"'$WD_ID'","status":"COMPLETED","rail_tx_id":"rail_payout_001"}' | jq

curl -L "http://localhost:8000/v1/wallets/$WALLET_ID/statement.csv" -o statement.csv
