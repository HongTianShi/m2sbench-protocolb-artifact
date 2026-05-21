from .common import build_reference, init_points, nearest_target_step
from ..core.affine_match import final_affine_correction

def run(dense_support, budget, n_points, seed=0, config=None):
    cfg = config or {}
    target = build_reference(dense_support, budget, n_points, seed=seed)
    pts = init_points(cfg.get("init_mode", "source"), dense_support, budget, n_points, seed=seed, target=target)
    lr = float(cfg.get("lr", 0.22))
    iters = int(cfg.get("iterations", 8))
    record_all = bool(cfg.get("record_all_intermediates", False))
    log_every = int(cfg.get("log_every", 1))
    intermediates = []
    for it in range(1, iters + 1):
        pts = pts + lr * (nearest_target_step(pts, target) - pts)
        pts = final_affine_correction(pts, budget["mean"], budget["covariance"])
        if record_all:
            if it % max(log_every, 1) == 0 or it == iters:
                intermediates.append({"iteration": it, "points": pts.tolist()})
        elif it in {1, max(2, iters // 2), iters}:
            intermediates.append({"iteration": it, "points": pts.tolist()})
    return {"points": pts, "intermediates": intermediates}
