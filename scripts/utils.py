from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> dict:
    with open(ROOT / 'config' / 'settings.yml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def p(*parts) -> Path:
    return ROOT.joinpath(*parts)


def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
