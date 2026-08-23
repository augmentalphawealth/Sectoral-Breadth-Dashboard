import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

files = [
    "security_master_with_kite.parquet",
    "nse_mainboard_master_bse_classified.parquet",
]

for file_name in files:
    file_path = PROCESSED / file_name

    print()
    print("=" * 70)
    print(file_name)
    print("=" * 70)

    if not file_path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_parquet(file_path)

    print(f"Rows: {len(df)}")
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))
