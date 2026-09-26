"""Parity and correctness tests for mojo-graphhopper-py.

There is no installable `graphhopper` Python package to compare against (see
the README), so the oracle here is deliberately independent of the code under
test: an O(V*E) Bellman-Ford with no heap, no levels and no shortcuts, plus
closed-form spherical geometry. A bug in the Mojo heap, in the level
comparison, in the witness search or in the shortcut expansion cannot hide
inside either of them.
"""

import ctypes
import math

import numpy as np
import pytest

import mojo_graphhopperpy as mgh
from mojo_graphhopperpy import _lib

R = 6371000.0


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def py_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (
        np.asarray(lat1, dtype=np.float64), np.asarray(lon1, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64), np.asarray(lon2, dtype=np.float64),
    )
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def test_haversine_quarter_meridian_is_exact():
    # Pole to equator is a quarter of the great circle, by definition.
    got = mgh.haversine([0.0], [0.0], [90.0], [0.0])
    np.testing.assert_allclose(got[0], R * math.pi / 2, rtol=1e-12)


def test_haversine_one_degree_at_equator():
    a, b, c, d = [0.0, 52.0], [0.0, 13.0], [0.0, 52.0], [1.0, 13.0]
    got = mgh.haversine(a, b, c, d)
    expect = py_haversine(np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d))
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    assert got[1] == 0.0  # identical points


def test_haversine_matches_numpy_reference():
    rng = np.random.default_rng(3)
    n = 500
    lat1 = rng.uniform(-89.0, 89.0, n)
    lon1 = rng.uniform(-180.0, 180.0, n)
    lat2 = rng.uniform(-89.0, 89.0, n)
    lon2 = rng.uniform(-180.0, 180.0, n)
    np.testing.assert_allclose(
        mgh.haversine(lat1, lon1, lat2, lon2),
        py_haversine(lat1, lon1, lat2, lon2),
        rtol=1e-12, atol=1e-6,
    )


def test_haversine_antipodal_is_half_circumference():
    a = np.array([0.0]); b = np.array([0.0])
    c = np.array([0.0]); d = np.array([180.0])
    got = mgh.haversine(a, b, c, d)
    np.testing.assert_allclose(got[0], math.pi * R, rtol=1e-12)


def test_bearing_cardinal_directions():
    lat1 = np.array([0.0, 0.0, 0.0, 45.0])
    lon1 = np.array([0.0, 0.0, 0.0, 0.0])
    lat2 = np.array([1.0, 0.0, 0.0, 46.0])
    lon2 = np.array([0.0, 1.0, 179.0, 0.0])
    got = mgh.bearing(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(got[0], 0.0, atol=1e-9)  # due north
    np.testing.assert_allclose(got[1], 90.0, atol=1e-9)  # due east
    np.testing.assert_allclose(got[2], 90.0, atol=1e-9)  # across the date line
    np.testing.assert_allclose(got[3], 0.0, atol=1e-9)  # due north


def test_equirectangular_matches_its_definition():
    lat1 = np.array([52.0, 0.0, 10.0])
    lon1 = np.array([13.0, 0.0, 20.0])
    lat2 = np.array([52.01, 0.0, 10.5])
    lon2 = np.array([13.02, 0.0, 21.0])
    mean = np.radians((lat1 + lat2) / 2)
    x = np.radians(lon2 - lon1) * np.cos(mean)
    y = np.radians(lat2 - lat1)
    np.testing.assert_allclose(
        mgh.equirectangular(lat1, lon1, lat2, lon2),
        R * np.hypot(x, y), rtol=1e-12,
    )


def test_equirectangular_agrees_with_haversine_for_short_hops():
    # The whole reason GraphHopper falls back to a plane projection.
    lat1 = np.full(50, 52.0)
    lon1 = np.full(50, 13.0)
    lat2 = lat1 + 0.001
    lon2 = lon1 + 0.001
    a = mgh.equirectangular(lat1, lon1, lat2, lon2)
    b = mgh.haversine(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(a, b, rtol=2e-3)


def test_edge_weights_are_length_over_speed():
    lat1 = np.array([52.0, 52.0])
    lon1 = np.array([13.0, 13.0])
    lat2 = np.array([52.01, 53.0])
    lon2 = np.array([13.0, 13.0])
    sp = np.array([10.0, 27.5])
    got = mgh.edge_weights(lat1, lon1, lat2, lon2, sp)
    expect = py_haversine(lat1, lon1, lat2, lon2) / sp
    np.testing.assert_allclose(got, expect, rtol=1e-12)


def test_edge_weights_reject_non_positive_speed():
    with pytest.raises(ValueError):
        mgh.edge_weights([0.0], [0.0], [1.0], [0.0], [0.0])


def test_geometry_rejects_ragged_input():
    with pytest.raises(ValueError):
        mgh.haversine([0.0, 1.0], [0.0], [0.0], [0.0])


def test_geometry_runs_on_buffers_above_four_gibibytes():
    """A ctypes address argument that gets truncated to 32 bits is silent.

    Every small-array test would still pass with a `c_int` argtype, because
    small heap allocations sit below 4 GiB; only a buffer above 4 GiB exposes
    the truncation. A fixed-address `mmap` is refused with EPERM in this
    sandbox, so this instead takes whatever address the allocator hands out for
    a large mapping and skips when that lands below the boundary.
    """
    from mojo_graphhopperpy import _lib

    big = np.zeros(160 * 1024 * 1024, dtype=np.uint8)  # 160 MiB, mmap'd
    try:
        if big.ctypes.data <= (1 << 32):
            pytest.skip("allocator did not place the buffer above 4 GiB")
        base = big.ctypes.data
        lat1 = np.ascontiguousarray([0.0, 0.0], dtype=np.float64)
        lon1 = np.ascontiguousarray([0.0, 0.0], dtype=np.float64)
        lat2 = np.ascontiguousarray([0.0, 1.0], dtype=np.float64)
        lon2 = np.ascontiguousarray([1.0, 1.0], dtype=np.float64)
        res = np.zeros(2, dtype=np.float64)
        off = base + (96 << 20)  # a page-aligned slot well inside the mapping
        ctypes.memmove(off, lat1.ctypes.data, lat1.nbytes)
        ctypes.memmove(off + 0x40, lon1.ctypes.data, lon1.nbytes)
        ctypes.memmove(off + 0x80, lat2.ctypes.data, lat2.nbytes)
        ctypes.memmove(off + 0xC0, lon2.ctypes.data, lon2.nbytes)
        ctypes.memmove(off + 0x100, res.ctypes.data, res.nbytes)
        args = _lib.Args()
        ctx = args.set([
            (_lib.G_LAT_A, off), (_lib.G_LON_A, off + 0x40),
            (_lib.G_LAT_B, off + 0x80), (_lib.G_LON_B, off + 0xC0),
            (_lib.G_OUT, off + 0x100),
        ])
        assert off > (1 << 32)
        assert all(args.buf[s] > (1 << 32) for s in (
            _lib.G_LAT_A, _lib.G_LON_A, _lib.G_LAT_B, _lib.G_LON_B, _lib.G_OUT,
        ))
        _lib.lib.gh_haversine(2, int(ctx))
        ctypes.memmove(res.ctypes.data, off + 0x100, res.nbytes)
        np.testing.assert_allclose(
            res,
            py_haversine([0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]),
            rtol=1e-12,
        )
    finally:
        del big


# --------------------------------------------------------------------------
# graphs and shortest paths
# --------------------------------------------------------------------------


def test_graph_rejects_bad_input():
    with pytest.raises(ValueError):
        mgh.Graph([0.0, 1.0], [0.0, 1.0], [(0, 5)])
    with pytest.raises(ValueError):
        mgh.Graph([0.0, 1.0], [0.0, 1.0], [(0, 0)])
    with pytest.raises(ValueError):
        mgh.Graph([0.0, 1.0], [0.0, 1.0], [(0, 1)], speeds=[-1.0])
    with pytest.raises(ValueError):
        mgh.Graph([], [], [])


def test_queries_reject_out_of_range_nodes():
    g = mgh.grid_graph(3, 3)
    with pytest.raises(IndexError):
        g.dijkstra(0, 99)
    with pytest.raises(IndexError):
        g.astar(-1, 0)
    with pytest.raises(IndexError):
        g.prepare_ch().route(0, 10 ** 9)


def test_path_to_self_is_a_single_node():
    g = mgh.grid_graph(4, 4)
    for p in (g.dijkstra(5, 5), g.astar(5, 5), g.prepare_ch().route(5, 5)):
        assert list(p.nodes) == [5]
        assert p.time == 0.0
        assert p.distance == 0.0


def test_path_points_match_node_coordinates():
    g = mgh.grid_graph(4, 4)
    p = g.dijkstra(0, 15)
    np.testing.assert_allclose(p.points[:, 0], g.lat[p.nodes])
    np.testing.assert_allclose(p.points[:, 1], g.lon[p.nodes])


@pytest.mark.parametrize("rows,cols,stride", [
    (2, 2, 0), (3, 5, 0), (4, 4, 2), (5, 3, 3), (6, 6, 0), (1, 6, 0),
])
def test_all_three_algorithms_agree_with_bellman_ford(rows, cols, stride):
    g = mgh.grid_graph(rows, cols, one_way_stride=stride)
    ch = g.prepare_ch()
    rng = np.random.default_rng(rows * 100 + cols * 10 + stride)
    for _ in range(20):
        s = int(rng.integers(g.node_count))
        t = int(rng.integers(g.node_count))
        _, ref = mgh.reference_shortest_path(g, s, t)
        for name, p in (
            ("dijkstra", g.dijkstra(s, t)),
            ("astar", g.astar(s, t)),
            ("ch", ch.route(s, t)),
        ):
            assert p is not None, f"{name} found no path {s}->{t}"
            assert list(p.nodes)[0] == s and list(p.nodes)[-1] == t, name
            assert p.time == pytest.approx(ref, rel=1e-9, abs=1e-9), name


def test_random_weighted_graphs_agree_with_bellman_ford():
    """Not a grid: irregular degree, random weights, no structure to exploit."""
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = int(rng.integers(20, 60))
        pts = rng.uniform(-89.0, 89.0, (n, 2))
        seen = set()
        edges = []
        for a in range(n):
            for _ in range(3):
                b = int(rng.integers(n))
                if a != b and (min(a, b), max(a, b)) not in seen:
                    seen.add((min(a, b), max(a, b)))
                    edges.append((a, b))
        speeds = rng.uniform(3.0, 30.0, len(edges))
        g = mgh.Graph(pts[:, 0], pts[:, 1], np.asarray(edges), speeds)
        ch = g.prepare_ch()
        for _ in range(8):
            s = int(rng.integers(n))
            t = int(rng.integers(n))
            _, ref = mgh.reference_shortest_path(g, s, t)
            got_d = g.dijkstra(s, t)
            got_a = g.astar(s, t)
            got_c = ch.route(s, t)
            if math.isinf(ref):
                assert got_d is None and got_a is None and got_c is None
                continue
            for name, p in (("dij", got_d), ("ast", got_a), ("ch", got_c)):
                assert p is not None, name
                assert p.time == pytest.approx(ref, rel=1e-9, abs=1e-9), name


def test_unreachable_pairs_return_none():
    """Two grids with no edge between them."""
    lat = np.array([52.0, 52.0, 53.0, 53.0])
    lon = np.array([13.0, 13.01, 14.0, 14.01])
    g = mgh.Graph(lat, lon, [(0, 1), (2, 3)])
    ch = g.prepare_ch()
    assert g.dijkstra(0, 2) is None
    assert g.astar(0, 2) is None
    assert ch.route(0, 2) is None
    assert g.dijkstra(0, 1) is not None
    assert ch.route(2, 3) is not None


def test_directed_graph_respects_one_ways():
    lat = np.array([52.0, 52.0, 52.0])
    lon = np.array([13.0, 13.01, 13.02])
    # 0 -> 1 -> 2 only
    g = mgh.Graph(lat, lon, [(0, 1), (1, 2)], directed=True)
    ch = g.prepare_ch()
    assert g.dijkstra(0, 2) is not None
    assert g.dijkstra(2, 0) is None
    assert ch.route(0, 2) is not None
    assert ch.route(2, 0) is None
    _, ref = mgh.reference_shortest_path(g, 0, 2)
    assert g.dijkstra(0, 2).time == pytest.approx(ref, rel=1e-12)


def test_speed_changes_the_optimal_path():
    """A slower direct edge must lose to a longer, faster detour."""
    lat = np.array([52.0, 52.0, 52.01, 52.01])
    lon = np.array([13.0, 13.02, 13.0, 13.02])
    edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
    fast = mgh.Graph(lat, lon, edges, [1.0, 1.0, 30.0, 30.0])
    slow = mgh.Graph(lat, lon, edges, [30.0, 30.0, 1.0, 1.0])
    # Speeds swap which of the two routes wins, and each result must match the
    # independent reference for its own speed assignment.
    a = fast.dijkstra(0, 3)
    b = slow.dijkstra(0, 3)
    assert a.time != pytest.approx(b.time, rel=1e-6)
    assert fast.astar(0, 3).time == pytest.approx(a.time, rel=1e-12)
    assert slow.astar(0, 3).time == pytest.approx(b.time, rel=1e-12)
    for g in (fast, slow):
        ch = g.prepare_ch()
        _, ref = mgh.reference_shortest_path(g, 0, 3)
        assert ch.route(0, 3).time == pytest.approx(ref, rel=1e-9)


def _python_astar(graph, src, dst, heur):
    """Independent A*, to pin down the settled set and not just the cost."""
    import heapq

    head, to, w = graph._head, graph._to, graph._weight
    dist = {src: 0.0}
    done = set()
    pq = [(heur[src], src)]
    while pq:
        _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            break
        for e in range(int(head[u]), int(head[u + 1])):
            v = int(to[e])
            ng = dist[u] + w[e]
            if ng < dist.get(v, math.inf):
                dist[v] = ng
                heapq.heappush(pq, (ng + heur[v], v))
    return done, dist.get(dst, math.inf)


def test_astar_settles_exactly_the_nodes_a_reference_astar_does():
    """Pins the heuristic plumbing, not just the answer.

    On a grid the great-circle bound is weak, so the settled set is large; what
    matters is that it is exactly the set an independent A* produces.
    """
    import heapq
    from mojo_graphhopperpy import _lib
    from mojo_graphhopperpy._lib import addr

    g = mgh.grid_graph(20, 20, slow_column_stride=10 ** 9)
    src, dst = 0, g.node_count - 1
    heur = np.ascontiguousarray(
        mgh.haversine(
            np.full(g.n, g.lat[src]), np.full(g.n, g.lon[src]), g.lat, g.lon
        ) / g.max_speed
    )
    want_set, want_cost = _python_astar(g, src, dst, heur)
    sc = g._scratch
    args = _lib.Args()
    ctx = args.set([
        (_lib.S_HEAD, addr(g._head)), (_lib.S_TO, addr(g._to)),
        (_lib.S_W, addr(g._weight)), (_lib.S_DIST, addr(sc.dist)),
        (_lib.S_PREV, addr(sc.prev)), (_lib.S_DONE, addr(sc.done)),
        (_lib.S_HK, addr(sc.hk)), (_lib.S_HV, addr(sc.hv)),
        (_lib.S_HS, addr(sc.hs)), (_lib.S_HEUR, addr(heur)),
    ])
    assert _lib.lib.gh_astar(g.n, src, dst, ctx) == 1
    assert {i for i in np.flatnonzero(sc.done)} == want_set
    assert sc.dist[dst] == pytest.approx(want_cost, rel=1e-12)
    assert g.astar(src, dst).time == pytest.approx(want_cost, rel=1e-12)


def test_astar_expansion_order_depends_on_the_heuristic():
    """A settled set that matches Dijkstra's would mean the heuristic is ignored.

    A great-circle bound is tight along a row of the grid, so A* and Dijkstra
    settle genuinely different sets there; the set must also match an
    independent A* exactly, which pins the ordering rather than just the cost.
    """
    from mojo_graphhopperpy import _lib
    from mojo_graphhopperpy._lib import addr

    g = mgh.grid_graph(20, 20, slow_column_stride=10 ** 9)
    src, dst = 0, 19
    heur = np.ascontiguousarray(
        mgh.haversine(
            np.full(g.n, g.lat[src]), np.full(g.n, g.lon[src]), g.lat, g.lon
        ) / g.max_speed
    )
    sc = g._scratch
    args = _lib.Args()
    ctx = args.set([
        (_lib.S_HEAD, addr(g._head)), (_lib.S_TO, addr(g._to)),
        (_lib.S_W, addr(g._weight)), (_lib.S_DIST, addr(sc.dist)),
        (_lib.S_PREV, addr(sc.prev)), (_lib.S_DONE, addr(sc.done)),
        (_lib.S_HK, addr(sc.hk)), (_lib.S_HV, addr(sc.hv)),
        (_lib.S_HS, addr(sc.hs)), (_lib.S_HEUR, addr(heur)),
    ])
    assert _lib.lib.gh_astar(g.n, src, dst, ctx) == 1
    astar_set = {int(i) for i in np.flatnonzero(sc.done)}
    want_set, want_cost = _python_astar(g, src, dst, heur)
    assert astar_set == want_set
    assert sc.dist[dst] == pytest.approx(want_cost, rel=1e-12)

    zeros = np.zeros(g.n)
    ctx0 = args.set([(_lib.S_HEUR, addr(zeros))])
    assert _lib.lib.gh_astar(g.n, src, dst, ctx0) == 1
    zero_set = {int(i) for i in np.flatnonzero(sc.done)}
    assert zero_set != astar_set
    dijkstra_set = _python_astar(g, src, dst, zeros)[0]
    assert zero_set == dijkstra_set
    assert g.astar(src, dst).time == pytest.approx(want_cost, rel=1e-12)


def test_contraction_hierarchy_builds_shortcuts():
    g = mgh.grid_graph(12, 12)
    ch = g.prepare_ch()
    assert ch.shortcuts > 0
    # the base pass seeds a forward and a reverse entry per directed CSR edge,
    # and star_entries() counts only the forward one
    assert ch.base_entries == 2 * g.edge_count
    assert ch.star_entries() == g.edge_count + ch.shortcuts
    assert ch.levels.max() > 0
    # every node except the untouched ones is contracted
    assert (ch.levels > 0).sum() > g.node_count // 2


def test_ch_preprocessing_retries_when_capacity_is_short():
    """A capacity too small for the shortcuts must grow, not silently truncate."""
    g = mgh.grid_graph(8, 8)
    tight = mgh.CHGraph(g, slack=0)
    roomy = mgh.CHGraph(g, slack=64)
    assert tight.shortcuts == roomy.shortcuts
    assert tight.levels.tolist() == roomy.levels.tolist()


def test_ch_result_is_a_real_base_graph_path():
    g = mgh.grid_graph(7, 7)
    ch = g.prepare_ch()
    p = ch.route(3, 45)
    for a, b in zip(p.nodes[:-1], p.nodes[1:]):
        assert g._edge_index(int(a), int(b)) >= 0
