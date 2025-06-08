from typing import Iterator, IO

import pandas as pd
def iter_excel_pandas(file: IO[bytes]) -> Iterator[dict[str, object]]:
    yield from pd.read_excel(file).to_dict('records')

if __name__ == "__main__":
    df = pd.read_excel('/Users/krishnaramachandran/pyexperiments/applications/llm_ap_ar_agent/mockdata/samples/john/timecard.xlsx', usecols='A,B,K,L')
    print(df.to_string())
