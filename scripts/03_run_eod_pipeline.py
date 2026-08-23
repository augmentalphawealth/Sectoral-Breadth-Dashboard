from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name)],
        check=True,
    )


def main() -> None:
    run("01_build_group_features.py")
    run("02_build_dashboard_tables.py")
    run("12_build_dashboard_history.py")
    run("13_build_dashboard_stock_history.py")

    sync_path = ROOT / "data" / "processed" / "last_sync.txt"
    sync_path.write_text(
        datetime.now().isoformat(),
        encoding="utf-8",
    )

    print("eod pipeline complete")


if __name__ == "__main__":
    main()
