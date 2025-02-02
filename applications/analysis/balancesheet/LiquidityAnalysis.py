import pandas as pd
def read_input_file() :
    try:
        df = pd.read_csv("/Users/krishnaramachandran/Downloads/earnings_master_1734823649749.csv")
        print(df.columns)
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")

if __name__ == '__main__' :
    read_input_file()
