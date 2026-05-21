from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILY_DIR = ROOT / "configs" / "families"

def load_family_catalog():
    return pd.read_csv(FAMILY_DIR / "family_catalog.csv")

def load_instance_catalog():
    return pd.read_csv(FAMILY_DIR / "instance_catalog.csv")

def list_instance_ids():
    return load_instance_catalog()["instance_id"].tolist()

def _seed_from_instance(instance_id: str) -> int:
    h = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % (2**32 - 1)

def _rng(instance_id: str):
    return np.random.default_rng(_seed_from_instance(instance_id))

def _difficulty_scale(difficulty: str):
    return {"easy": 0.8, "medium": 1.0, "hard": 1.25}.get(str(difficulty), 1.0)

def materialize_instance(instance_id: str, n_dense: int = 1200) -> dict:
    cat = load_instance_catalog()
    row = cat[cat["instance_id"] == instance_id]
    if len(row) == 0:
        raise KeyError(f"Unknown instance_id: {instance_id}")
    row = row.iloc[0].to_dict()
    fam = row["family"]
    diff = row.get("difficulty", "medium")
    rng = _rng(instance_id)
    s = _difficulty_scale(diff)
    t = np.linspace(0.0, 1.0, n_dense)

    if fam == "trend":
        x = t
        y = 0.15 * np.sin(6 * np.pi * t) + (0.7 + 0.2 * s) * t
    elif fam == "bubble-crash":
        x = t
        peak = np.exp(-((t - 0.58) / (0.10 / s)) ** 2)
        y = 0.25 * t + 0.9 * peak - 0.9 * np.maximum(t - 0.62, 0) * s * 2.0
    elif fam == "regime-shift":
        x = t
        y = 0.25 * np.sin(4 * np.pi * t) + np.where(t < 0.5, -0.35, 0.45) + 0.08 * s * np.sign(t - 0.5)
    elif fam == "volatility-clustering":
        x = t
        y = (0.05 + 0.22 * (np.sin(5 * np.pi * t) > 0)) * rng.normal(size=n_dense)
        y = np.convolve(y, np.ones(5) / 5, mode="same")
    elif fam == "mean-reversion":
        x = t
        y = 0.5 * np.sin((8 + 2 * s) * np.pi * t) * np.exp(-0.4 * t)
    elif fam == "star":
        theta = np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
        r = 0.5 + 0.25 * np.sign(np.sin(5 * theta))
        x = r * np.cos(theta); y = r * np.sin(theta)
    elif fam == "spiral":
        theta = np.linspace(0.2, 4 * np.pi, n_dense)
        r = np.linspace(0.05, 0.8, n_dense)
        x = r * np.cos(theta); y = r * np.sin(theta)
    elif fam == "double-loop":
        theta = np.linspace(0, 2 * np.pi, n_dense)
        x = np.sin(theta); y = np.sin(theta) * np.cos(theta)
    elif fam == "ring-hole":
        theta = np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
        r = 0.75 + 0.02 * rng.normal(size=n_dense)
        x = r * np.cos(theta); y = r * np.sin(theta)
    elif fam == "s-curve":
        x = np.linspace(-1, 1, n_dense); y = np.tanh(2.2 * x)
    else:
        x = np.linspace(-1, 1, n_dense); y = np.zeros_like(x)

    pts = np.column_stack([x, y]).astype(float)
    noise = 0.01 * s * rng.normal(size=pts.shape)
    pts = pts + noise
    return {
        "instance_id": instance_id,
        "family": fam,
        "domain": row.get("domain", "unknown"),
        "difficulty": diff,
        "points": pts,
        "seed": int(row.get("seed", _seed_from_instance(instance_id))),
    }
