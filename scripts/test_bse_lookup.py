from bse import BSE


TEST_ISIN = "INE040A01034"   # HDFCBANK
TEST_SYMBOL = "HDFCBANK"


def main():
    bse = BSE()

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
