from __future__ import annotations

import random
from typing import Any


def solve(cell: dict[str, Any]) -> dict[str, Any]:
    support = cell["granted_views"]["support_points"]
    n_points = int(cell["n_points"])
    rng = random.Random(cell["cell_id"])
    if len(support) >= n_points:
        candidate = rng.sample(support, n_points)
    else:
        candidate = [rng.choice(support) for _ in range(n_points)]
    ranked_views = ["moments", "occupancy", "raster", "point"]
    return {
        "cell_id": cell["cell_id"],
        "method": "random_support_adapter",
        "candidate": candidate,
        "ranked_views": ranked_views,
        "route": "moments",
    }
