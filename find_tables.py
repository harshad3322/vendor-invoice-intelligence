import sqlite3
import pandas as pd

conn = sqlite3.connect("data/inventory.db")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print(tables)

conn.close()