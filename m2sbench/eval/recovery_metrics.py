import numpy as np

def _chamfer(a, b):
    da = ((a[:,None,:]-b[None,:,:])**2).sum(axis=2)
    return float(da.min(axis=1).mean() + da.min(axis=0).mean())

def _coverage(pred, target, radius=None):
    d = ((pred[:,None,:]-target[None,:,:])**2).sum(axis=2)
    min_d = np.sqrt(d.min(axis=0))
    if radius is None:
        td = np.sqrt(((target[:,None,:]-target[None,:,:])**2).sum(axis=2) + np.eye(len(target))*1e9)
        radius = float(np.quantile(td.min(axis=1), 0.5))
    return float((min_d <= radius).mean())

def _raster(points, size=64):
    pts = np.asarray(points, float)
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-8)
    xy = ((pts - mn) / span * (size - 1)).astype(int)
    grid = np.zeros((size, size), dtype=np.uint8)
    xy = np.clip(xy, 0, size - 1)
    grid[xy[:,1], xy[:,0]] = 1
    return grid

def _iou(a, b):
    ra, rb = _raster(a), _raster(b)
    inter = np.logical_and(ra, rb).sum()
    union = np.logical_or(ra, rb).sum()
    return float(inter / union) if union else 1.0

def _cov(x):
    return np.cov(np.asarray(x).T, bias=True)

def _uniformity(points):
    pts=np.asarray(points,float)
    d=np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(axis=2)+np.eye(len(pts))*1e9)
    nn=d.min(axis=1)
    mu=float(nn.mean())
    return float(nn.std()/mu) if mu>0 else 0.0

def evaluate_recovery(pred_points, target_points, budget=None):
    pred=np.asarray(pred_points,float); tgt=np.asarray(target_points,float)
    norm = np.sqrt(np.trace(_cov(tgt))) + 1e-8
    cd = _chamfer(pred, tgt) / norm
    cov = _coverage(pred, tgt)
    iou = _iou(pred, tgt)
    uni = _uniformity(pred)
    stat = 0.0
    if budget is not None:
        mu = np.asarray(budget["mean"], float)
        covm = np.asarray(budget["covariance"], float)
        stat = max(float(np.abs(pred.mean(axis=0)-mu).max()), float(np.abs(_cov(pred)-covm).max()))
    return {
        "norm_chamfer": float(cd),
        "coverage": float(cov),
        "raster_iou": float(iou),
        "max_stat_error": float(stat),
        "uniformity_cv": float(uni),
    }
