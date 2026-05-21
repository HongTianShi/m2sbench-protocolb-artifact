import numpy as np

def farthest_point_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = min(n, len(points))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(points)))
    selected = [first]
    dists = np.full(len(points), np.inf)
    for _ in range(1, n):
        last = points[selected[-1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return points[selected]
