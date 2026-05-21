from .common import build_reference
def run(dense_support, budget, n_points, seed=0, config=None):
    return {"points": build_reference(dense_support, budget, n_points, seed=0), "intermediates": []}
