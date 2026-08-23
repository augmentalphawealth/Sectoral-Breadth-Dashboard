from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master.parquet"


def main():
    df = pd.read_parquet(INPUT_FILE)

    print("========== SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMNS ==========")
    for column in df.columns:
        print(column)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head().to_string(index=False))

    print("\n========== DTYPES ==========")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
