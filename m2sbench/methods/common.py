import numpy as np
from ..core.canonical_target import construct_canonical_target
from ..core.affine_match import final_affine_correction
from ..core.sampling import farthest_point_sample

def build_reference(dense_support, budget, n_points, seed=0):
    return construct_canonical_target(dense_support, n_points, budget["mean"], budget["covariance"], seed=seed)

def source_template(n_points, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_points)
    x = 2.0 * t - 1.0
    y = 0.45 * np.sin(3.0 * np.pi * t) + 0.12 * np.sign(np.sin(7.0 * np.pi * t))
    pts = np.column_stack([x, y])
    pts += 0.01 * rng.normal(size=pts.shape)
    return pts

def init_points(mode, dense_support, budget, n_points, seed=0, target=None):
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        pts = rng.normal(size=(n_points, 2))
    elif mode == "source":
        pts = source_template(n_points, seed=seed)
    elif mode == "support-sampled":
        pts = farthest_point_sample(dense_support, n_points, seed=seed)
    elif mode == "support-jitter":
        pts = farthest_point_sample(dense_support, n_points, seed=seed) + 0.03 * rng.normal(size=(n_points, 2))
    elif mode == "target-jitter":
        if target is None:
            raise ValueError("target-jitter init requires target")
        pts = np.asarray(target, float) + 0.02 * rng.normal(size=np.asarray(target).shape)
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    return final_affine_correction(pts, budget["mean"], budget["covariance"])

def nearest_target_step(points, target):
    pts = np.asarray(points, float); tgt = np.asarray(target, float)
    d2 = ((pts[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    return tgt[idx]

def assignment_step(points, target):
    pts = np.asarray(points, float); tgt = np.asarray(target, float)
    d2 = ((pts[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d2, axis=1)
    return tgt[idx]

def sinkhorn_barycenters(points, target, epsilon=0.08):
    pts = np.asarray(points, float); tgt = np.asarray(target, float)
    d2 = ((pts[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / max(epsilon, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs @ tgt

def repulsion_force(points, weight=0.005):
    pts = np.asarray(points, float)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(axis=2) + np.eye(len(pts))
    inv = 1.0 / np.maximum(d2, 1e-6)
    force = (diff * inv[:, :, None]).sum(axis=1)
    return weight * force / max(len(pts), 1)
