"""Interleaved A/B timing of two builds of the same package.

Alternates the two library paths round by round instead of running one build
to completion and then the other, so a slow minute on the shared host lands on
both columns as evenly as possible.

Usage::

    pixi run python bench/ab.py SERIAL_DIR PARALLEL_DIR [--reps N]

Each DIR is a checkout containing `dist/libmojo-graphhopper-py.so` and
`python/mojo_graphhopperpy`. The report is best-of-N wall clock per build, the
ratio, and a correctness check that both builds return the same geometry.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import time

import numpy as np


def load(root):
    """Import the package from `root`, replacing any previously loaded copy.

    `_lib` resolves its `.so` from the package file's own path, so swapping the
    entry ahead of the importer is enough to switch builds.
    """
    entry = str(pathlib.Path(root).resolve() / "python")
    for name in [m for m in list(sys.modules) if m.startswith("mojo_graphhopperpy")]:
        del sys.modules[name]
    if sys.path[0] == entry:
        del sys.path[0]
    sys.path.insert(0, entry)
    return importlib.import_module("mojo_graphhopperpy")


def timeit(fn, repeats):
    t = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serial")
    ap.add_argument("parallel")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    a = load(args.serial)
    b = load(args.parallel)

    n = 1 << 20
    rng = np.random.default_rng(0)
    coords = tuple(rng.uniform(lo, hi, n) for lo, hi in
                   ((-89.0, 89.0), (-180.0, 180.0), (-89.0, 89.0), (-180.0, 180.0)))
    sp = rng.uniform(1.0, 30.0, n)

    cases = {
        "haversine n=1M": (lambda m: m.haversine(*coords), a.haversine(*coords)),
        "equirectangular n=1M": (lambda m: m.equirectangular(*coords), a.equirectangular(*coords)),
        "bearing n=1M": (lambda m: m.bearing(*coords), a.bearing(*coords)),
        "edge_weights n=1M": (lambda m: m.edge_weights(*coords, sp), a.edge_weights(*coords, sp)),
        "haversine n=2^18": (
            lambda m: m.haversine(*[c[:1 << 18] for c in coords]),
            a.haversine(*[c[:1 << 18] for c in coords]),
        ),
        "haversine n=65536": (
            lambda m: m.haversine(*[c[:65536] for c in coords]),
            a.haversine(*[c[:65536] for c in coords]),
        ),
    }

    # correctness gate: same answer from both builds
    for label, (fn, ref) in cases.items():
        got = fn(b)
        assert np.array_equal(got, ref), f"{label}: builds disagree"

    per_case = {k: ([], []) for k in cases}
    for _ in range(args.reps):
        for label, (fn, _) in cases.items():
            per_case[label][0].append(timeit(lambda: fn(a), 1))
            per_case[label][1].append(timeit(lambda: fn(b), 1))

    print(f"{'case':<28}{'serial':>12}{'parallel':>12}{'before/after':>15}")
    print("-" * 67)
    for label, (ta, tb) in per_case.items():
        sa, sb = min(ta), min(tb)
        print(f"{label:<28}{sa * 1e3:>10.2f}ms{sb * 1e3:>10.2f}ms{sa / sb:>13.2f}x")


if __name__ == "__main__":
    main()
