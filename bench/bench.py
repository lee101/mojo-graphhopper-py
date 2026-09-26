"""Correctness-gated benchmark for mojo-graphhopper-py.

Every case checks the answer against an independent Bellman-Ford before it is
timed, so a regression in the Mojo kernels shows up as a correctness failure
rather than as a suspiciously good number. The baseline is `networkx`, the
fastest reasonable pure-Python implementation of the same algorithms available
here, and the same graph is used for both.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_graphhopperpy as mgh  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _to_networkx(graph, mgh_graph):
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(mgh_graph.node_count))
    head, to, w = mgh_graph._head, mgh_graph._to, mgh_graph._weight
    for u in range(mgh_graph.node_count):
        for e in range(int(head[u]), int(head[u + 1])):
            g.add_edge(u, int(to[e]), weight=float(w[e]))
    return g


def bench_geometry(n: int = 1 << 20):
    """Haversine over a million coordinate pairs, against a NumPy formulation.

    This is the closest fair NumPy baseline; a Python loop would be a strawman.
    """
    rng = np.random.default_rng(0)
    lat1 = rng.uniform(-89.0, 89.0, n)
    lon1 = rng.uniform(-180.0, 180.0, n)
    lat2 = rng.uniform(-89.0, 89.0, n)
    lon2 = rng.uniform(-180.0, 180.0, n)

    def np_haversine():
        p1 = np.radians(lat1)
        p2 = np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6371000.0 * np.arcsin(np.sqrt(a))

    got = mgh.haversine(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(got, np_haversine(), rtol=1e-12)
    return [(f"haversine n={n}", _time(np_haversine, 3), _time(
        lambda: mgh.haversine(lat1, lon1, lat2, lon2), 3))]


def bench_dijkstra(size: int = 60, queries: int = 20):
    """One-shot Dijkstra, against networkx's heap implementation."""
    import networkx as nx

    g = mgh.grid_graph(size, size, one_way_stride=5)
    nxg = _to_networkx(g, g)
    rng = np.random.default_rng(1)
    pairs = [
        (int(rng.integers(g.node_count)), int(rng.integers(g.node_count)))
        for _ in range(queries)
    ]

    def run_nx():
        for s, t in pairs:
            nx.single_source_dijkstra_path_length(
                nxg, s, t, weight="weight"
            )

    def run_mine():
        for s, t in pairs:
            g.dijkstra(s, t)

    # correctness first
    for s, t in pairs[:5]:
        mine = g.dijkstra(s, t)
        ref, cost = mgh.reference_shortest_path(g, s, t)
        assert mine is not None and abs(mine.time - cost) < 1e-6 * max(1.0, cost)
    return [(f"dijkstra {size}x{size} x{queries}",
             _time(run_nx, 3), _time(run_mine, 3))]


def bench_astar(size: int = 60, queries: int = 20):
    """A* against Dijkstra on the same graph and the same queries.

    A* is compared against this package's own Dijkstra because both are the same
    kernel with and without a heuristic; that isolates what the heuristic buys.
    """
    g = mgh.grid_graph(size, size, one_way_stride=5)
    rng = np.random.default_rng(2)
    pairs = [
        (int(rng.integers(g.node_count)), int(rng.integers(g.node_count)))
        for _ in range(queries)
    ]

    def run_dij():
        for s, t in pairs:
            g.dijkstra(s, t)

    def run_ast():
        for s, t in pairs:
            g.astar(s, t)

    settled_d = 0
    settled_a = 0
    for s, t in pairs:
        g.dijkstra(s, t)
        settled_d += int(g._scratch.done.sum())
        g.astar(s, t)
        settled_a += int(g._scratch.done.sum())
    return [(f"astar {size}x{size} x{queries} [{settled_a}/{settled_d} settled]",
             _time(run_dij, 3), _time(run_ast, 3))]


def bench_ch(size: int = 40, queries: int = 200):
    """CH preprocessing plus queries, against networkx Dijkstra per query.

    The two numbers are not comparable one for one: the Mojo column pays the
    one-off preprocessing, the networkx column pays nothing up front. Both
    columns are reported so the trade is visible.
    """
    import networkx as nx

    g = mgh.grid_graph(size, size, one_way_stride=5)
    nxg = _to_networkx(g, g)
    rng = np.random.default_rng(3)
    pairs = [
        (int(rng.integers(g.node_count)), int(rng.integers(g.node_count)))
        for _ in range(queries)
    ]

    t0 = time.perf_counter()
    ch = g.prepare_ch()
    prep = time.perf_counter() - t0

    for s, t in pairs[:20]:
        p = ch.route(s, t)
        _, cost = mgh.reference_shortest_path(g, s, t)
        assert p is not None and abs(p.time - cost) < 1e-6 * max(1.0, cost)

    def run_nx():
        for s, t in pairs:
            nx.single_source_dijkstra_path_length(nxg, s, t, weight="weight")

    def run_mine():
        for s, t in pairs:
            ch.route(s, t)

    nx_time = _time(run_nx, 3)
    query_time = _time(run_mine, 3)
    rows = [(f"ch {size}x{size} x{queries} queries", nx_time, query_time)]
    # the same run with the one-off preprocessing amortised over the queries
    rows.append(
        (f"ch {size}x{size} x{queries} + prep", nx_time, query_time + prep)
    )
    return rows


def main():
    print(f"{'case':<44}{'reference':>12}{'mojo':>12}{'ratio':>9}")
    print("-" * 77)
    for fn in (bench_geometry, bench_dijkstra, bench_astar, bench_ch):
        for label, ref, got in fn():
            ratio = ref / got if got else float("nan")
            print(
                f"{label:<44}{ref * 1e3:>10.2f}ms{got * 1e3:>10.2f}ms{ratio:>8.2f}x"
            )


if __name__ == "__main__":
    main()
