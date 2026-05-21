from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BUDGET_DIR = ROOT / "configs" / "budgets"

def list_budgets():
    return sorted([p.stem for p in BUDGET_DIR.glob("*.yaml")])

def load_budget(name: str) -> dict:
    path = BUDGET_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown budget '{name}'. Expected {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["budget_name"] = data.get("budget_name", name)
    return data
