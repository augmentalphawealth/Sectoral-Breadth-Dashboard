from pathlib import Path

from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = ROOT / "data" / "bse_test_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BSE_CODE = "500180"  # HDFCBANK


def main():
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        print("Testing equityMetaInfo for BSE code:", BSE_CODE)
        meta = bse.equityMetaInfo(BSE_CODE)

        print("\n========== EQUITY META INFO ==========")
        print(meta)
        print("\n========== TYPE ==========")
        print(type(meta))

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
