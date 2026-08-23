from pathlib import Path

from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = ROOT / "data" / "bse_test_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def main():
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        print("Available BSE methods (starting with 'get' or 'fetch' or 'search' or 'lookup'):")
        methods = [m for m in dir(bse) if not m.startswith("_")]

        keywords = ["get", "fetch", "search", "lookup", "info", "company", "industry", "sector"]
        filtered = [m for m in methods if any(k in m.lower() for k in keywords)]

        for m in sorted(filtered):
            print(m)

        print("\n========== ALL METHODS ==========")
        for m in sorted(methods):
            print(m)

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
