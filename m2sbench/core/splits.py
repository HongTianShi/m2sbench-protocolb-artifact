from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "configs" / "splits"

def load_split(name: str):
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def validate_split(df: pd.DataFrame):
    return {"rows": len(df), "columns": list(df.columns)}
