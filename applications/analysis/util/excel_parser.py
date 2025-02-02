import sys
import pandas as pd
from io import StringIO

excel_file = pd.ExcelFile(sys.argv[1])
for sheet_name in excel_file.sheet_names:
    print('Sheet: %s' % sheet_name)
    output = StringIO()
    output2 = StringIO()
    output2.write('Sheet: %s' % sheet_name)
    pd.read_excel(sys.argv[1], sheet_name=sheet_name).to_csv(output, header=False, index=True)
    print(output.getvalue())