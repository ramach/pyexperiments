# main.py
from agents.llm_ap_ar_agent import create_financial_agent, run_agent_query
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Query to run", required=True)
    args = parser.parse_args()

    agent = create_financial_agent()
    response = run_agent_query(agent, args.query)
    print("\nResponse:\n", response)
