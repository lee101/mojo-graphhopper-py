"""Per-kernel breakdown of the routing benchmark, for profiling a change.

`bench/bench.py` reports whole-operation numbers. This runs the same graphs and
the same query lists at the same sizes and times each kernel on its own, so a
change to one loop can be attributed to it instead of to the sum. It also times
the Python-side path finish, which is part of `dijkstra`/`astar`/`route` wall
clock and none of the kernels can move.

Usage::

    pixi run python bench/kernels.py [-t] [-c]

`-t` is the quick configuration used while iterating (smaller query counts,
fewer repeats); the defaults are the `bench/bench.py` sizes. `-c` takes no bench
lock, for a box where another run of the shared lock is in progress.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_graphhopperpy as mgh  # noqa: E402
from mojo_graphhopperpy import _lib  # noqa: E402


def best(fn, repeats):
    out = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", action="store_true", help="quick configuration")
    ap.add_argument("-c", action="store_true", help="informational only: no lock")
    args = ap.parse_args()
    reps = 3 if args.t else 5
    n_q = 60 if args.t else 200
    lib = _lib.lib

    rows = []

    # --- geometry -----------------------------------------------------------
    n = 1 << 20
    rng = np.random.default_rng(0)
    lat1 = rng.uniform(-89.0, 89.0, n)
    lon1 = rng.uniform(-180.0, 180.0, n)
    lat2 = rng.uniform(-89.0, 89.0, n)
    lon2 = rng.uniform(-180.0, 180.0, n)
    f_lat1, f_lon1, f_lat2, f_lon2 = (
        _lib.f64(lat1), _lib.f64(lon1), _lib.f64(lat2), _lib.f64(lon2),
    )
    res = np.empty(n, dtype=np.float64)
    ctx = _lib.Args()
    ctx.set([
        (_lib.G_LAT_A, _lib.addr(f_lat1)), (_lib.G_LON_A, _lib.addr(f_lon1)),
        (_lib.G_LAT_B, _lib.addr(f_lat2)), (_lib.G_LON_B, _lib.addr(f_lon2)),
        (_lib.G_OUT, _lib.addr(res)),
    ])
    addr = int(ctx)
    rows.append(("haversine kernel n=1M", best(lambda: lib.gh_haversine(n, addr), reps)))
    rows.append(("haversine wrapper n=1M",
                 best(lambda: mgh.haversine(lat1, lon1, lat2, lon2), reps)))

    # --- dijkstra / astar ---------------------------------------------------
    g = mgh.grid_graph(60, 60, one_way_stride=5)
    qrng = np.random.default_rng(1)
    pairs = [
        (int(qrng.integers(g.node_count)), int(qrng.integers(g.node_count)))
        for _ in range(n_q)
    ]
    sc = g._scratch
    caddr = sc.fill(g)
    rows.append(("dijkstra 60x60 kernel x%d" % n_q,
                 best(lambda: [lib.gh_dijkstra(g.n, s, t, caddr) for s, t in pairs], reps)))
    rows.append(("dijkstra 60x60 full x%d" % n_q,
                 best(lambda: [g.dijkstra(s, t) for s, t in pairs], reps)))

    # A* needs a heuristic per source, which the wrapper recomputes per query;
    # isolate the search kernel with a fixed heuristic so the geometry cost of
    # the heuristic is not mixed in.
    g._heur_cache = np.zeros(g.n, dtype=np.float64)
    aaddr = sc.fill(g, with_heur=True)
    rows.append(("astar 60x60 search-only x%d" % n_q,
                 best(lambda: [lib.gh_astar(g.n, s, t, aaddr) for s, t in pairs], reps)))
    rows.append(("astar 60x60 full x%d" % n_q,
                 best(lambda: [g.astar(s, t) for s, t in pairs], reps)))

    # the Python half of a query: predecessor walk plus path finish
    # sc.prev is whatever the last kernel call left behind, so re-run the search
    # that produced it rather than reading a buffer from a different pair.
    def python_half():
        for s, t in pairs[:20]:
            if lib.gh_dijkstra(g.n, s, t, caddr) == 1:
                g._finish(g._path_from_prev(sc.prev, s, t))
    rows.append(("path_from_prev+finish x20", best(python_half, reps)))

    # --- CH -----------------------------------------------------------------
    size = 40
    chg = mgh.grid_graph(size, size, one_way_stride=5)
    prep = best(lambda: chg.prepare_ch(), 1)
    rows.append((f"ch {size}x{size} prepare", prep))
    ch = chg.prepare_ch()
    crng = np.random.default_rng(3)
    cpairs = [
        (int(crng.integers(chg.node_count)), int(crng.integers(chg.node_count)))
        for _ in range(n_q)
    ]
    g2 = ch.graph
    mid_ctx = ch._mid_args.set([
        (_lib.C_FHEAD, _lib.addr(ch._fhead)), (_lib.C_FNXT, _lib.addr(ch._fnxt)),
        (_lib.C_FTO, _lib.addr(ch._fto)), (_lib.C_FW, _lib.addr(ch._fw)),
        (_lib.C_RHEAD, _lib.addr(ch._rhead)), (_lib.C_RNXT, _lib.addr(ch._rnxt)),
        (_lib.C_RTO, _lib.addr(ch._rto)), (_lib.C_RW, _lib.addr(ch._rw)),
        (_lib.C_LEVEL, _lib.addr(ch.levels)),
        (_lib.M_DF, _lib.addr(ch._df)), (_lib.M_DT, _lib.addr(ch._dt)),
        (_lib.M_SF, _lib.addr(ch._sf)), (_lib.M_ST, _lib.addr(ch._st)),
        (_lib.M_HK, _lib.addr(ch._hk)), (_lib.M_HV, _lib.addr(ch._hv)),
        (_lib.M_HS, _lib.addr(ch._hs)),
    ])
    rows.append(("ch middle down+up x%d" % n_q,
                 best(lambda: [lib.gh_ch_middle(g2.n, s, t, mid_ctx) for s, t in cpairs], reps)))

    down_ctx = ch._down_args.set([
        (_lib.C_FHEAD, _lib.addr(ch._fhead)), (_lib.C_FNXT, _lib.addr(ch._fnxt)),
        (_lib.C_FTO, _lib.addr(ch._fto)), (_lib.C_FW, _lib.addr(ch._fw)),
        (_lib.C_FMID, _lib.addr(ch._fmid)),
        (_lib.C_RHEAD, _lib.addr(ch._rhead)), (_lib.C_RNXT, _lib.addr(ch._rnxt)),
        (_lib.C_RTO, _lib.addr(ch._rto)), (_lib.C_RW, _lib.addr(ch._rw)),
        (_lib.C_RMID, _lib.addr(ch._rmid)),
        (_lib.C_LEVEL, _lib.addr(ch.levels)),
        (_lib.S_DIST, _lib.addr(ch._dist)), (_lib.S_PREV, _lib.addr(ch._prev_edge)),
        (_lib.S_PREVN, _lib.addr(ch._prev_node)), (_lib.S_DONE, _lib.addr(ch._done)),
        (_lib.S_HK, _lib.addr(ch._hk)), (_lib.S_HV, _lib.addr(ch._hv)),
        (_lib.S_HS, _lib.addr(ch._hs)),
    ])
    middles = [lib.gh_ch_middle(g2.n, s, t, mid_ctx) for s, t in cpairs]
    mids = [(c, m) for c, m in zip(cpairs, middles) if m >= 0]
    for rev in (1, 0):
        rows.append((
            "ch downward rev=%d x%d" % (rev, len(mids)),
            best(lambda rev=rev: [lib.gh_ch_downward(g2.n, m, rev, down_ctx) for _, m in mids], reps),
        ))
    rows.append(("ch route x%d (query time)" % n_q,
                 best(lambda: [ch.route(s, t) for s, t in cpairs], reps)))

    # --- report -------------------------------------------------------------
    print(f"{'kernel':<38}{'best':>12}")
    print("-" * 50)
    for label, t in rows:
        print(f"{label:<38}{t * 1e3:>10.2f}ms")


if __name__ == "__main__":
    main()
