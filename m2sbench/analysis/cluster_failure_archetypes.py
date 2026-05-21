import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def summarize_run(d):
    d = d.sort_values("iteration")
    first = d.iloc[0]
    last = d.iloc[-1]
    return {
        "method": last["method"],
        "budget_name": last["budget_name"],
        "family": last["family"],
        "seed": int(last["seed"]),
        "start_cd": float(first["norm_chamfer"]),
        "end_cd": float(last["norm_chamfer"]),
        "delta_cd": float(last["norm_chamfer"] - first["norm_chamfer"]),
        "end_coverage": float(last["coverage"]),
        "end_topology": float(last["topology_raw"]),
        "end_collapse": float(last["collapse_raw"]),
        "n_steps": int(len(d)),
    }

def main():
    ap = argparse.ArgumentParser(description="Cluster representative failure trajectory archetypes.")
    ap.add_argument("--trajectory_runs", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(args.trajectory_runs)
    runs = []
    for keys, d in traj.groupby(["method","budget_name","family","seed"]):
        runs.append(summarize_run(d))
    df = pd.DataFrame(runs)
    feat_cols = ["start_cd","end_cd","delta_cd","end_coverage","end_topology","end_collapse","n_steps"]
    X = df[feat_cols].values
    k = min(4, len(df))
    km = KMeans(n_clusters=k, random_state=0, n_init=20)
    df["archetype"] = km.fit_predict(X)
    emb = PCA(n_components=2, random_state=0).fit_transform(X)
    df["pca1"] = emb[:,0]; df["pca2"] = emb[:,1]
    df.to_csv(out / "failure_archetypes.csv", index=False)
    (out / "failure_archetypes_summary.json").write_text(json.dumps({
        "n_runs": int(len(df)),
        "n_archetypes": int(k),
        "feature_columns": feat_cols,
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
