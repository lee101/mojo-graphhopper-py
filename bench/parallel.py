"""Thread scaling for the parallel paths, and the serial/parallel crossover.

`bench/bench.py` and `bench/kernels.py` report one configuration each. This
script compares Mojo against Mojo -- one build, one input, the serial path
against the parallel path -- which is the only fair way to decide whether the
parallel path is worth shipping. `networkx` is not required.

The kernels read `GH_WORKERS` on every call. The reader is clamped to
`PAR_WORKER_CAP` (4) and cannot be raised above it. `GH_WORKERS=1` makes the
caller take the serial loop, which is also what a caller below the size
threshold gets -- so switching the variable between measurements of the same
workload is exactly the 1/2/4-thread comparison, done in one process on one
graph, with nothing else moving.

Every case is checked before it is timed: against the NumPy formulation for
geometry, against the Bellman-Ford oracle for the routing kernels.

Usage::

    pixi run python bench/parallel.py            # thread sweep, full cases
    pixi run python bench/parallel.py -t         # quick: fewer repeats
    pixi run python bench/parallel.py --sizes     # serial/parallel crossover

The host is shared. Numbers are best-of-N wall clock; compare within one run.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_graphhopperpy as mgh  # noqa: E402
from mojo_graphhopperpy import _lib  # noqa: E402

CAP = 4  # mirrors PAR_WORKER_CAP in src/kernels.mojo
COUNTS = (1, 2, CAP)


def best(fn, repeats):
    out = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def at_workers(count):
    """Context manager setting GH_WORKERS for one measurement and restoring it."""
    class _Ctx:
        def __enter__(self):
            self.old = os.environ.get("GH_WORKERS")
            os.environ["GH_WORKERS"] = str(count)

        def __exit__(self, *exc):
            if self.old is None:
                os.environ.pop("GH_WORKERS", None)
            else:
                os.environ["GH_WORKERS"] = self.old

    return _Ctx()


def geometry(n):
    rng = np.random.default_rng(0)
    return tuple(rng.uniform(lo, hi, n) for lo, hi in
                 ((-89.0, 89.0), (-180.0, 180.0), (-89.0, 89.0), (-180.0, 180.0)))


def np_haversine(lat_a, lon_a, lat_b, lon_b):
    p1, p2 = np.radians(lat_a), np.radians(lat_b)
    dp = np.radians(lat_b - lat_a)
    dl = np.radians(lon_b - lon_a)
    x = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(x))


def measure(fn, repeats):
    """Best time per worker count; returns {count: seconds}."""
    out = {}
    for w in (1, 2, CAP):
        with at_workers(w):
            out[w] = best(fn, repeats)
    return out


def report(rows, repeats):
    print(f"{'case':<34}{'1 thread':>12}{'2 threads':>12}{'4 threads':>12}"
          f"{'2x':>7}{'4x':>7}")
    print("-" * 84)
    for label, fn in rows:
        t = measure(fn, repeats)
        print(f"{label:<34}{t[1] * 1e3:>10.2f}ms{t[2] * 1e3:>10.2f}ms"
              f"{t[CAP] * 1e3:>10.2f}ms{t[1] / t[2]:>6.2f}x{t[1] / t[CAP]:>6.2f}x")


def size_sweep(repeats):
    """Where does parallel start to beat serial, for the geometry kernels?"""
    print(f"{'n':>10}{'serial':>13}{'cap=4':>13}{'ratio':>8}")
    print("-" * 44)
    for n in (1 << 12, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20):
        coords = geometry(n)
        with at_workers(1):
            serial = best(lambda: mgh.haversine(*coords), repeats)
        with at_workers(CAP):
            par = best(lambda: mgh.haversine(*coords), repeats)
        print(f"{n:>10}{serial * 1e3:>11.3f}ms{par * 1e3:>11.3f}ms"
              f"{serial / par:>7.2f}x")
    os.environ.pop("GH_WORKERS", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", action="store_true", help="quick: fewer repeats")
    ap.add_argument("--sizes", action="store_true", help="serial/parallel crossover")
    args = ap.parse_args()
    reps = 3 if args.t else 7

    if args.sizes:
        size_sweep(reps)
        return

    coords = geometry(1 << 20)
    lat_a, lon_a, lat_b, lon_b = coords
    assert np.allclose(mgh.haversine(*coords), np_haversine(*coords), rtol=1e-12)

    g = mgh.grid_graph(60, 60, one_way_stride=5)
    rng = np.random.default_rng(1)
    pairs = [(int(rng.integers(g.n)), int(rng.integers(g.n))) for _ in range(200)]
    for s, t in pairs[:5]:
        p = g.dijkstra(s, t)
        _, cost = mgh.reference_shortest_path(g, s, t)
        assert p is not None and abs(p.time - cost) <= 1e-6 * max(1.0, cost)
    ch = g.prepare_ch()
    for s, t in pairs[:5]:
        p = ch.route(s, t)
        _, cost = mgh.reference_shortest_path(g, s, t)
        assert p is not None and abs(p.time - cost) <= 1e-6 * max(1.0, cost)

    rows = [
        ("haversine n=1M", lambda: mgh.haversine(*coords)),
        ("equirectangular n=1M", lambda: mgh.equirectangular(*coords)),
        ("bearing n=1M", lambda: mgh.bearing(*coords)),
        ("ch 60x60 prepare", lambda: g.prepare_ch()),
        ("ch 60x60 route x200", lambda: [ch.route(s, t) for s, t in pairs]),
        ("dijkstra 60x60 x200", lambda: [g.dijkstra(s, t) for s, t in pairs]),
    ]
    report(rows, reps)


if __name__ == "__main__":
    main()
