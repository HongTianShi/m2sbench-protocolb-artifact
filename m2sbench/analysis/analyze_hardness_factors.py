import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from m2sbench.core.budgets import load_budget
from m2sbench.core.families import materialize_instance, load_instance_catalog

def structural_features(points):
    pts = np.asarray(points, float)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    dx = np.diff(pts[:, 0]); dy = np.diff(pts[:, 1])
    slopes = dy / np.maximum(np.abs(dx), 1e-6)
    curvature = np.diff(slopes)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    area = float(bbox[0] * bbox[1])
    aspect = float(bbox[0] / max(bbox[1], 1e-8))
    turn_count = int((np.sign(slopes[1:]) != np.sign(slopes[:-1])).sum()) if len(slopes) > 1 else 0
    roughness = float(np.std(curvature)) if len(curvature) else 0.0
    length = float(np.sqrt(dx**2 + dy**2).sum())
    yfreq = float(np.mean(np.abs(np.diff(np.sign(dy)) > 0))) if len(dy) > 1 else 0.0
    return {
        "bbox_area": area,
        "bbox_aspect": aspect,
        "path_length": length,
        "slope_std": float(np.std(slopes)) if len(slopes) else 0.0,
        "curvature_std": roughness,
        "turn_count": float(turn_count),
        "x_span": float(bbox[0]),
        "y_span": float(bbox[1]),
        "density": float(len(pts) / max(area, 1e-6)),
    }

def budget_features(budget_name):
    b = load_budget(budget_name)
    cov = np.asarray(b["covariance"], float)
    eig = np.linalg.eigvalsh(cov)
    return {
        "budget_trace": float(np.trace(cov)),
        "budget_det": float(np.linalg.det(cov)),
        "budget_condition": float(eig.max() / max(eig.min(), 1e-8)),
        "budget_anisotropy": float((eig.max() - eig.min()) / max(eig.max(), 1e-8)),
    }

def main():
    ap = argparse.ArgumentParser(description="Exploratory hardness modeling over Track 1 cells.")
    ap.add_argument("--track1_best_iterative", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(args.track1_best_iterative)
    rows = []
    for _, r in best.iterrows():
        inst = materialize_instance(f"{r['target_name']}_medium_001")
        feats = {}
        feats.update(structural_features(inst["points"]))
        feats.update(budget_features(r["reference_family"]))
        feats["target_name"] = r["target_name"]
        feats["target_family"] = r["target_family"]
        feats["reference_family"] = r["reference_family"]
        feats["ambiguity_severity"] = float(r["ambiguity_severity"])
        rows.append(feats)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["target_name","target_family","reference_family","ambiguity_severity"]]
    X = df[feature_cols].values
    y = df["ambiguity_severity"].values

    model = RandomForestRegressor(n_estimators=300, random_state=0)
    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=0)
    imp = pd.DataFrame({"feature": feature_cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    pred = df[["target_name","target_family","reference_family","ambiguity_severity"]].copy()
    pred["predicted_severity"] = y_pred
    pred["abs_error"] = np.abs(pred["predicted_severity"] - pred["ambiguity_severity"])
    summary = {
        "n_cells": int(len(df)),
        "cv_mae": float(np.mean(np.abs(y_pred - y))),
        "cv_rmse": float(np.sqrt(np.mean((y_pred - y)**2))),
        "corr_observed_predicted": float(np.corrcoef(y, y_pred)[0, 1]) if len(y) > 1 else 1.0,
    }
    df.to_csv(out / "hardness_feature_matrix.csv", index=False)
    imp.to_csv(out / "hardness_feature_importance.csv", index=False)
    pred.to_csv(out / "hardness_predictions.csv", index=False)
    (out / "hardness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
