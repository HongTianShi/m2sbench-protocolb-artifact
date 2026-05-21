from pathlib import Path
import yaml
from . import constructive, direct_optimization, distance_field, sinkhorn_guided, hard_projection

METHOD_REGISTRY = {
    "constructive": constructive.run,
    "direct-optimization": direct_optimization.run,
    "distance-field": distance_field.run,
    "sinkhorn-guided": sinkhorn_guided.run,
    "hard-projection": hard_projection.run,
}
METHOD_NAMES = sorted(METHOD_REGISTRY.keys())

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "methods"

def load_method_config(method_name: str, profile: str = "main-paper") -> dict:
    key = method_name.replace("-", "_")
    path = CONFIG_DIR / f"{key}.yaml"
    if not path.exists():
        return {"method_name": method_name, "profile": profile, "config_name": f"{key}:{profile}"}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    cfg = profiles.get(profile, {}).copy()
    cfg["method_name"] = method_name
    cfg["profile"] = profile
    cfg["config_name"] = f"{key}:{profile}"
    return cfg

def run_method(method: str, dense_support, budget: dict, n_points: int, seed: int = 0, config: dict | None = None):
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method}")
    cfg = config or load_method_config(method)
    return METHOD_REGISTRY[method](dense_support, budget, n_points, seed=seed, config=cfg)
