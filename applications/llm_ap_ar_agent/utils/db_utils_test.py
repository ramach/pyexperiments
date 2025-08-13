import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_util_postgres_2 import init_db, insert_correction, get_last_correction
load_dotenv()
init_db()
insert_correction("ACME Corp", "total_amount", "", "$2000", "Added missing value")
print(get_last_correction("ACME Corp", "total_amount"))
