import time
from contextlib import contextmanager

@contextmanager
def timed():
    t0 = time.perf_counter()
    out = {}
    yield out
    out["runtime_s"] = time.perf_counter() - t0
