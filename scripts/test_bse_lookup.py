from pathlib import Path

from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = ROOT / "data" / "bse_test_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BSE_CODE = "500180"  # HDFCBANK


def main():
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        print("Testing BSE get_company_info...")
        info = bse.get_company_info(BSE_CODE)

        print("\n========== COMPANY INFO ==========")
        print(info)
        print("\n========== TYPE ==========")
        print(type(info))

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
