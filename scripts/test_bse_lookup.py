from pathlib import Path

from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = ROOT / "data" / "bse_test_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

TEST_ISIN = "INE040A01034"   # HDFCBANK
TEST_SYMBOL = "HDFCBANK"


def main():
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        print("Testing BSE lookup by ISIN...")
        isin_result = bse.lookup(TEST_ISIN)

        print("\n========== ISIN RESULT ==========")
        print(isin_result)
        print("\n========== ISIN RESULT TYPE ==========")
        print(type(isin_result))

        print("\nTesting BSE lookup by symbol...")
        symbol_result = bse.lookup(TEST_SYMBOL)

        print("\n========== SYMBOL RESULT ==========")
        print(symbol_result)
        print("\n========== SYMBOL RESULT TYPE ==========")
        print(type(symbol_result))

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
