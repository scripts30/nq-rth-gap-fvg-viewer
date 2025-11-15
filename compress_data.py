import os
import pandas as pd

DATA_DIR = "data"
CSV_NAME = "historical_1m.csv"
GZ_NAME = "historical_1m.csv.gz"

csv_path = os.path.join(DATA_DIR, CSV_NAME)
gz_path = os.path.join(DATA_DIR, GZ_NAME)

print(f"Reading full CSV: {csv_path}")
df = pd.read_csv(csv_path)
print("Shape:", df.shape)

print(f"Writing compressed CSV: {gz_path}")
df.to_csv(gz_path, index=False, compression="gzip")

print("Done.")
