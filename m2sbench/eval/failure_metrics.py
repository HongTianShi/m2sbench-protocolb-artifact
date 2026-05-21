import numpy as np

def _raster(pts, size=64):
    pts=np.asarray(pts,float)
    mn=pts.min(axis=0); mx=pts.max(axis=0); span=np.maximum(mx-mn, 1e-8)
    xy=((pts-mn)/span*(size-1)).astype(int)
    grid=np.zeros((size,size), dtype=np.uint8)
    xy=np.clip(xy, 0, size-1)
    grid[xy[:,1], xy[:,0]]=1
    return grid

def evaluate_failure(pred_points, target_points, source_points=None, recovery_metrics=None):
    pred=np.asarray(pred_points,float); tgt=np.asarray(target_points,float)
    rm = recovery_metrics or {}
    coverage = float(rm.get("coverage", 0.0))
    uniformity = float(rm.get("uniformity_cv", 0.0))
    collapse_raw = max(0.0, (1.0 - coverage) * max(uniformity, 0.0))
    topology_raw = float(np.logical_xor(_raster(pred), _raster(tgt)).sum() / max(np.logical_or(_raster(pred), _raster(tgt)).sum(),1))
    locking_raw = 0.0
    locking_defined = False
    if source_points is not None and len(source_points):
        src=np.asarray(source_points,float)
        d_src=((pred[:,None,:]-src[None,:,:])**2).sum(axis=2).min(axis=1).mean()
        d_tgt=((pred[:,None,:]-tgt[None,:,:])**2).sum(axis=2).min(axis=1).mean()
        locking_raw=max(0.0, float(d_tgt - d_src))
        locking_defined=True
    vals = {
        "locking_raw": float(locking_raw),
        "collapse_raw": float(collapse_raw),
        "topology_raw": float(topology_raw),
        "locking_score": float(locking_raw),
        "collapse_score": float(collapse_raw),
        "topology_score": float(topology_raw),
        "locking_defined": bool(locking_defined),
    }
    dom = max({"locking":vals["locking_score"],"collapse":vals["collapse_score"],"topology":vals["topology_score"]}, key=lambda k: {"locking":vals["locking_score"],"collapse":vals["collapse_score"],"topology":vals["topology_score"]}[k])
    vals["dominant_failure"] = dom
    return vals
