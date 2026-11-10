"""Serial and parallel paths must return the same answer on the same input.

`src/kernels.mojo` reads `GH_WORKERS` on every geometry call and treats `1` as
"stay serial" -- the same path a caller below the size threshold takes. Each
test here runs one kernel twice on one input, once with the parallel path
forced off and once with it on, and requires either bit-identical output or
agreement inside the kernel's documented tolerance.

The parallel path is only entered at or above the size threshold, so the sizes
below are chosen to cross it: a small input documents that below-threshold
callers are unaffected, and the threshold itself and above exercise the chunk
split, including the non-divisible remainder that makes the last chunk ragged.
"""

import os

import numpy as np
import pytest

import mojo_graphhopperpy as mgh

# Mirrors PAR_MIN_ITEMS in src/kernels.mojo: below this, `num_workers` returns 0
# and the serial loop runs whatever GH_WORKERS says.
PAR_MIN_ITEMS = 262144


def with_workers(count):
    """Context manager setting GH_WORKERS for one call and restoring it."""

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


def coords(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-89.0, 89.0, n),
        rng.uniform(-180.0, 180.0, n),
        rng.uniform(-89.0, 89.0, n),
        rng.uniform(-180.0, 180.0, n),
        rng.uniform(1.0, 30.0, n),
    )


GEOMETRY = {
    "haversine": lambda c: mgh.haversine(c[0], c[1], c[2], c[3]),
    "equirectangular": lambda c: mgh.equirectangular(c[0], c[1], c[2], c[3]),
    "bearing": lambda c: mgh.bearing(c[0], c[1], c[2], c[3]),
    "edge_weights": lambda c: mgh.edge_weights(c[0], c[1], c[2], c[3], c[4]),
}


@pytest.mark.parametrize("name", sorted(GEOMETRY))
@pytest.mark.parametrize("n", [1000, PAR_MIN_ITEMS, PAR_MIN_ITEMS + 7, 1 << 20])
def test_geometry_serial_equals_parallel(name, n):
    """Bit-identical, not merely close: the chunk worker runs the same body."""
    fn = GEOMETRY[name]
    data = coords(n)
    with with_workers(1):
        serial = fn(data)
    with with_workers(4):
        par = fn(data)
    assert par.dtype == serial.dtype and par.shape == serial.shape
    assert np.array_equal(par, serial, equal_nan=True), (
        f"{name} n={n}: parallel output differs from serial"
    )


@pytest.mark.parametrize("name", sorted(GEOMETRY))
def test_geometry_parallel_intermediate_worker_counts(name):
    """2 workers must agree with 4; a 4-vs-2 difference would mean a race."""
    fn = GEOMETRY[name]
    data = coords(PAR_MIN_ITEMS + 1, seed=3)
    out = {}
    for w in (1, 2, 3, 4):
        with with_workers(w):
            out[w] = fn(data)
    for w in (2, 3, 4):
        assert np.array_equal(out[w], out[1], equal_nan=True), (
            f"{name}: {w} workers differs from serial"
        )


def test_threshold_boundary_matches_serial():
    """One item below the threshold and one item at it agree bit for bit."""
    below = coords(PAR_MIN_ITEMS - 1, seed=7)
    at = coords(PAR_MIN_ITEMS, seed=7)
    with with_workers(1):
        ref_below = mgh.haversine(*below[:4])
    with with_workers(4):
        got_below = mgh.haversine(*below[:4])
        got_at = mgh.haversine(*at[:4])
    with with_workers(1):
        ref_at = mgh.haversine(*at[:4])
    assert np.array_equal(got_below, ref_below)
    assert np.array_equal(got_at, ref_at)


def test_workers_env_cannot_exceed_cap():
    """The cap is a contract: a larger GH_WORKERS must not change the answer."""
    data = coords(PAR_MIN_ITEMS + 13, seed=11)
    with with_workers(4):
        capped = mgh.haversine(*data[:4])
    with with_workers(64):
        asked = mgh.haversine(*data[:4])
    assert np.array_equal(asked, capped)


def test_routing_queries_unaffected_by_worker_env():
    """Query kernels are serial; the geometry they call must not change results."""
    g = mgh.grid_graph(30, 30, one_way_stride=5)
    rng = np.random.default_rng(2)
    pairs = [(int(rng.integers(g.n)), int(rng.integers(g.n))) for _ in range(12)]
    with with_workers(1):
        a = [(g.dijkstra(s, t), g.astar(s, t)) for s, t in pairs]
    with with_workers(4):
        b = [(g.dijkstra(s, t), g.astar(s, t)) for s, t in pairs]
    for (p1, q1), (p2, q2) in zip(a, b):
        assert (p1 is None) == (p2 is None) and (q1 is None) == (q2 is None)
        if p1 is not None:
            assert np.array_equal(p1.nodes, p2.nodes)
            assert p1.distance == p2.distance and p1.time == p2.time
        if q1 is not None:
            assert np.array_equal(q1.nodes, q2.nodes)
            assert q1.distance == q2.distance and q1.time == q2.time
