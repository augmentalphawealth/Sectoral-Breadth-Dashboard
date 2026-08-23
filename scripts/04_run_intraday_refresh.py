from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / '02_build_dashboard_tables.py')], check=True)
    sync_path = ROOT / 'data' / 'processed' / 'last_sync.txt'
    sync_path.write_text(datetime.now().isoformat(), encoding='utf-8')
    print('intraday refresh complete')


if __name__ == '__main__':
    main()
