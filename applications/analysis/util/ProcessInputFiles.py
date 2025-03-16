import pandas as pd

from TestDriver import create_fintech_analyst2

pd.set_option('display.max_columns', None)
def read_input():
    df_assets = pd.read_csv("/Users/krishnaramachandran/Downloads/balance_sheet_assets_aal.csv.csv", sep=",")
    df_equity = pd.read_csv("/Users/krishnaramachandran/Downloads/balance_sheet_equity.csv.csv", sep=",")
    df_liabilities = pd.read_csv("/Users/krishnaramachandran/Downloads/balance_sheet_liabilities_aal.csv.csv", sep=",")
    df_income_statement0 = pd.read_csv("/Users/krishnaramachandran/Downloads/income_statement_aal.csv.csv", sep=",")
    df_income_statement = df_income_statement0[['act_symbol', 'net_income', 'income_from_continuing_operations']]
    df_merged0 = pd.merge(df_assets, df_liabilities, on='act_symbol', how='left')
    df_merged1 = pd.merge(df_merged0, df_equity, on='act_symbol', how='left')
    df_merged2 = pd.merge(df_merged1, df_income_statement, on='act_symbol', how='left')
    df_unused = df_merged2[['act_symbol', 'date', 'total_current_assets', 'total_current_liabilities',
             'inventories', 'cash_and_equivalents', 'net_income', 'income_from_continuing_operations']]
    data = pd.read_csv("/Users/krishnaramachandran/Downloads/merged_balance_income_aapl/part-00000-721bdb43-34db-43b2-bf7f-f29351d77850-c000.csv", sep=",")
    json_file_path = '/Users/krishnaramachandran/Downloads/merged_balance_income_aapl_json/input.json'
    data.to_json(json_file_path, orient='records', indent=4)
    df = pd.DataFrame(data)
    df['Working Capital'] = df['total_current_assets'] - df['total_current_liabilities']
    df['Current Ratio'] = df['total_current_assets'] / df['total_current_liabilities']
    df['Quick Ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liabilities']
    df['Cash Ratio'] = df['cash_and_equivalents'] / df['total_current_liabilities']
    df['Operating Cash Flow Ratio'] = df['income_from_continuing_operations'] / df['total_current_liabilities']
    df['Cash Flow to Revenue Ratio'] = df['income_from_continuing_operations'] / df['net_income']
    pd.set_option('display.max_columns', None)
    print(df.columns)
    return df
def main():
    df = read_input()
    fintech_analyst = (create_fintech_analyst2(df))
    # Example prompts from treasury and CFO
    prompts = [
        "What is the company's current ratio?",
        "How has the company's cash flow changed over the past year?",
        "What are the key liquidity risks facing the company?",
        "Can you provide a summary of the company's financial performance?",
        "How has the company's liquidity position changed over this period?",
        "Are there any concerning trends?",
        "What recommendations can you provide for improving liquidity management?",
    ]
    # Get answers from the agent
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = fintech_analyst.run(prompt)
        print(f"Response: {response}\n")
if __name__ == "__main__":
    main()


