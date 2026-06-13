import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("data/inventory.db")

# Read first 500 rows from vendor_invoice table
df = pd.read_sql(
    "SELECT * FROM vendor_invoice LIMIT 500",
    conn
)

# Save sample CSV
df.to_csv(
    "data/sample_invoice_data.csv",
    index=False
)

conn.close()

print("Sample dataset created successfully!")
print("Rows:", len(df))