"""Routing graph and query layer.

The Python side owns every array: the CSR graph, the search scratch, the
contraction-hierarchy forward stars and all their buffers. Mojo owns the loops
that actually touch them -- the heap, the relaxations, the contraction order,
the witness searches and the upward/downward searches.

Edge weights follow GraphHopper's `DistanceWeighting`: an edge weighs its
great-circle length divided by the speed on it. Without speeds the weight is the
length in metres, which is what makes the A* heuristic `haversine(src, v)` an
admissible and consistent lower bound.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from . import _lib
from ._lib import (
    C_DU,
    C_DW,
    C_FHEAD,
    C_FMID,
    C_FNXT,
    C_FTO,
    C_FW,
    C_LEVEL,
    C_MAXLVL,
    C_NBCNT,
    C_OHS,
    C_OHK,
    C_OHV,
    C_OCOUNT,
    C_ORDER,
    C_RHEAD,
    C_RMID,
    C_RNXT,
    C_RTO,
    C_RW,
    C_STAMPU,
    C_STAMPW,
    C_USED,
    C_WHS,
    C_WHK,
    C_WHV,
    G_LAT_A,
    G_LAT_B,
    G_LON_A,
    G_LON_B,
    G_OUT,
    G_SPEED,
    M_DF,
    M_DT,
    M_HS,
    M_HK,
    M_HV,
    M_SF,
    M_ST,
    S_DIST,
    S_DONE,
    S_HEAD,
    S_HEUR,
    S_HS,
    S_HK,
    S_HV,
    S_PREV,
    S_PREVN,
    S_TO,
    S_W,
    Args,
    addr,
    f64,
    lib,
)


BIG = 1.7976931348623157e308
EARTH_RADIUS_M = 6371000.0


def _geom(fn, lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = f64(lat1), f64(lon1), f64(lat2), f64(lon2)
    n = lat1.size
    for a in (lon1, lat2, lon2):
        if a.size != n:
            raise ValueError("latitude/longitude arrays must be the same length")
    res = np.empty(n, dtype=np.float64)
    ctx = Args()
    ctx.set([
        (G_LAT_A, addr(lat1)), (G_LON_A, addr(lon1)),
        (G_LAT_B, addr(lat2)), (G_LON_B, addr(lon2)),
        (G_OUT, addr(res)),
    ])
    fn(n, int(ctx))
    return res


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres for each elementwise pair."""
    return _geom(lib.gh_haversine, lat1, lon1, lat2, lon2)


def equirectangular(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Equirectangular (plane projection) distance in metres per pair."""
    return _geom(lib.gh_equirect, lat1, lon1, lat2, lon2)


def bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing in degrees in [0, 360) per pair."""
    return _geom(lib.gh_bearing, lat1, lon1, lat2, lon2)


def edge_weights(lat1, lon1, lat2, lon2, speeds) -> np.ndarray:
    """Edge weight: great-circle metres divided by `speeds` (metres/second)."""
    lat1, lon1, lat2, lon2, sp = (
        f64(lat1), f64(lon1), f64(lat2), f64(lon2), f64(speeds),
    )
    n = lat1.size
    if sp.size != n:
        raise ValueError("one speed per point pair is required")
    if n and float(sp.min()) <= 0.0:
        raise ValueError("speeds must be positive")
    res = np.empty(n, dtype=np.float64)
    ctx = Args()
    ctx.set([
        (G_LAT_A, addr(lat1)), (G_LON_A, addr(lon1)),
        (G_LAT_B, addr(lat2)), (G_LON_B, addr(lon2)),
        (G_OUT, addr(res)), (G_SPEED, addr(sp)),
    ])
    lib.gh_edge_weights(n, int(ctx))
    return res

class Path(NamedTuple):
    """A route, the way GraphHopper's `Path` reports one."""

    nodes: np.ndarray
    distance: float
    time: float
    points: np.ndarray


class _Scratch:
    """Search scratch owned by one graph, reused across queries."""

    def __init__(self, n: int, m: int):
        self.dist = np.empty(n, dtype=np.float64)
        self.prev = np.empty(n, dtype=np.int32)
        self.done = np.empty(n, dtype=np.uint8)
        # A push happens once for the source plus once per successful
        # relaxation, and each directed edge is relaxed at most once from each
        # end, so m + 8 slots can never overflow.
        cap = m + 8
        self.hk = np.empty(cap, dtype=np.float64)
        self.hv = np.empty(cap, dtype=np.int32)
        self.hs = np.empty(1, dtype=np.int64)
        self.args = Args()

    def fill(self, graph: "Graph", with_heur: bool = False) -> int:
        pairs = [
            (S_HEAD, addr(graph._head)),
            (S_TO, addr(graph._to)),
            (S_W, addr(graph._weight)),
            (S_DIST, addr(self.dist)),
            (S_PREV, addr(self.prev)),
            (S_DONE, addr(self.done)),
            (S_HK, addr(self.hk)),
            (S_HV, addr(self.hv)),
            (S_HS, addr(self.hs)),
        ]
        if with_heur:
            pairs.append((S_HEUR, addr(graph._heur_cache)))
        return self.args.set(pairs)


class Graph:
    """A CSR routing graph over nodes with coordinates.

    `edges` is an (m, 2) integer array of node index pairs; `speeds` is an (m,)
    array of metres per second. Without speeds the edge weight is the length in
    metres.
    """

    def __init__(
        self,
        lat: Sequence[float],
        lon: Sequence[float],
        edges,
        speeds: Optional[Sequence[float]] = None,
        directed: bool = False,
    ):
        self.lat = f64(lat)
        self.lon = f64(lon)
        n = self.lat.size
        if self.lon.size != n:
            raise ValueError("lat and lon must have the same length")
        if n == 0:
            raise ValueError("a graph needs at least one node")

        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        u = edges[:, 0]
        v = edges[:, 1]
        if u.size and (u.min() < 0 or v.min() < 0 or u.max() >= n or v.max() >= n):
            raise ValueError("edge endpoint out of range")
        if np.any(u == v):
            raise ValueError("self loops are not routable")
        if speeds is None:
            sp = np.ones(u.size, dtype=np.float64)
        else:
            sp = f64(speeds).reshape(-1)
            if sp.size != u.size:
                raise ValueError("one speed per edge is required")
            if np.any(sp <= 0.0):
                raise ValueError("speeds must be positive")
        self.speed = sp
        self.directed = bool(directed)
        self.max_speed = float(sp.max()) if sp.size else 1.0

        length = haversine(self.lat[u], self.lon[u], self.lat[v], self.lon[v])
        weight = length / sp

        if not directed:
            src = np.concatenate([u, v])
            dst = np.concatenate([v, u])
            length = np.concatenate([length, length])
            weight = np.concatenate([weight, weight])
        else:
            src, dst = u, v

        deg = np.bincount(src, minlength=n).astype(np.int64)
        head = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=head[1:])
        order = np.argsort(src, kind="stable")
        self._head = np.ascontiguousarray(head, dtype=np.int64)
        self._to = np.ascontiguousarray(dst[order], dtype=np.int32)
        self._length = np.ascontiguousarray(length[order], dtype=np.float64)
        self._weight = np.ascontiguousarray(weight[order], dtype=np.float64)
        self.n = n
        self.m = int(src.size)
        self._scratch = _Scratch(n, self.m)
        self._heur_cache = np.empty(n, dtype=np.float64)

    @property
    def node_count(self) -> int:
        return self.n

    @property
    def edge_count(self) -> int:
        return self.m

    def coordinates(self) -> np.ndarray:
        return np.column_stack([self.lat, self.lon])

    def _check_nodes(self, src: int, dst: int):
        for name, v in (("src", src), ("dst", dst)):
            if not isinstance(v, (int, np.integer)) or not 0 <= v < self.n:
                raise IndexError(f"{name} node {v} is not in the graph")

    def _edge_index(self, u: int, v: int) -> int:
        """Cheapest-weight directed CSR entry u->v, or -1."""
        best = -1
        best_w = BIG
        head = self._head
        for e in range(int(head[u]), int(head[u + 1])):
            if int(self._to[e]) == v and self._weight[e] < best_w:
                best_w = self._weight[e]
                best = e
        return best

    def _path_from_prev(self, prev: np.ndarray, start: int, end: int) -> np.ndarray:
        """Walk `prev` from `end` back to `start`, then reverse.

        The kernels store the predecessor of each node as "the node nearer the
        root of the search", so the chain runs against the direction of travel.
        """
        out = [int(end)]
        cur = int(end)
        for _ in range(self.n + 1):
            if cur == start:
                break
            cur = int(prev[cur])
            if cur < 0:
                raise RuntimeError(
                    f"no predecessor chain from {end} back to {start}"
                )
            out.append(cur)
        else:
            raise RuntimeError("search produced a cyclic predecessor chain")
        out.reverse()
        return np.asarray(out, dtype=np.int64)

    def _finish(self, nodes: np.ndarray) -> Path:
        total_len = 0.0
        total_w = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            e = self._edge_index(int(a), int(b))
            if e < 0:
                raise RuntimeError(f"path step {a}->{b} is not an edge")
            total_len += float(self._length[e])
            total_w += float(self._weight[e])
        pts = np.column_stack([self.lat[nodes], self.lon[nodes]])
        return Path(nodes=nodes, distance=total_len, time=total_w, points=pts)

    def dijkstra(self, src: int, dst: int) -> Optional[Path]:
        """Shortest path by weight. Returns None when `dst` is unreachable."""
        self._check_nodes(src, dst)
        ctx = self._scratch.fill(self)
        rc = lib.gh_dijkstra(self.n, int(src), int(dst), ctx)
        if rc != 1:
            return None
        return self._finish(self._path_from_prev(self._scratch.prev, src, dst))

    def astar(self, src: int, dst: int) -> Optional[Path]:
        """A* with a great-circle heuristic. Returns None when unreachable.

        The heuristic is `haversine(src, v) / max(speed)`, a lower bound on the
        time to reach `v` because no edge is faster than `max(speed)` and no
        edge is shorter than the great-circle distance between its ends.
        """
        self._check_nodes(src, dst)
        lat1 = np.full(self.n, float(self.lat[src]), dtype=np.float64)
        lon1 = np.full(self.n, float(self.lon[src]), dtype=np.float64)
        heur = haversine(lat1, lon1, self.lat, self.lon) / self.max_speed
        self._heur_cache = np.ascontiguousarray(heur, dtype=np.float64)
        ctx = self._scratch.fill(self, with_heur=True)
        rc = lib.gh_astar(self.n, int(src), int(dst), ctx)
        if rc != 1:
            return None
        return self._finish(self._path_from_prev(self._scratch.prev, src, dst))

    def prepare_ch(self) -> "CHGraph":
        """Preprocess contraction hierarchies for this graph."""
        return CHGraph(self)


class CHGraph:
    """A contraction-hierarchy query graph: GraphHopper's
    `PrepareContractionHierarchies` preprocessing plus its `PathBidirCH`
    style upward-search / downward-search query."""

    def __init__(self, graph: Graph, slack: int = 8):
        self.graph = graph
        n, m = graph.n, graph.m
        level = np.empty(n, dtype=np.int32)
        maxlvl = np.empty(n, dtype=np.int32)
        nbcnt = np.empty(n, dtype=np.int32)
        du = np.empty(n, dtype=np.float64)
        dw = np.empty(n, dtype=np.float64)
        stampu = np.empty(n, dtype=np.int32)
        stampw = np.empty(n, dtype=np.int32)
        # out-neighbours, in-neighbours, and two witness pop-lists
        order = np.empty(4 * n, dtype=np.int32)
        ocount = np.empty(1, dtype=np.int32)
        used = np.empty(3, dtype=np.int32)
        dist = np.empty(n, dtype=np.float64)
        prev = np.empty(n, dtype=np.int32)
        prev_node = np.empty(n, dtype=np.int32)
        done = np.empty(n, dtype=np.uint8)
        df = np.empty(n, dtype=np.float64)
        dt = np.empty(n, dtype=np.float64)
        sf = np.empty(n, dtype=np.int32)
        st = np.empty(n, dtype=np.int32)
        # The query heap shares the same bound as the plain searches: one push
        # per source plus one per successful relaxation.
        hcap = m + 8
        hk = np.empty(hcap, dtype=np.float64)
        hv = np.empty(hcap, dtype=np.int32)
        hs = np.empty(1, dtype=np.int64)

        cap = m + slack * n + 8
        while True:
            fhead = np.empty(n, dtype=np.int32)
            fnxt = np.empty(cap, dtype=np.int32)
            fto = np.empty(cap, dtype=np.int32)
            fw = np.empty(cap, dtype=np.float64)
            fmid = np.empty(cap, dtype=np.int32)
            rhead = np.empty(n, dtype=np.int32)
            rnxt = np.empty(cap, dtype=np.int32)
            rto = np.empty(cap, dtype=np.int32)
            rw = np.empty(cap, dtype=np.float64)
            rmid = np.empty(cap, dtype=np.int32)
            used[1] = cap
            # The contraction-order heap pushes once per node plus once per
            # (contracted node, uncontracted neighbour) pair, and the sum of
            # the forward out-degrees over the whole run is the star entry
            # count, which is at most `cap`.
            oheap_cap = n + cap + 8
            ohk = np.empty(oheap_cap, dtype=np.float64)
            ohv = np.empty(oheap_cap, dtype=np.int32)
            ohs = np.empty(1, dtype=np.int64)
            # One witness search relaxes each star entry at most once.
            whk = np.empty(cap + 8, dtype=np.float64)
            whv = np.empty(cap + 8, dtype=np.int32)
            whs = np.empty(1, dtype=np.int64)

            args = Args()
            ctx = args.set([
                (S_HEAD, addr(graph._head)), (S_TO, addr(graph._to)),
                (S_W, addr(graph._weight)),
                (C_FHEAD, addr(fhead)), (C_FNXT, addr(fnxt)),
                (C_FTO, addr(fto)), (C_FW, addr(fw)), (C_FMID, addr(fmid)),
                (C_RHEAD, addr(rhead)), (C_RNXT, addr(rnxt)),
                (C_RTO, addr(rto)), (C_RW, addr(rw)), (C_RMID, addr(rmid)),
                (C_USED, addr(used)), (C_LEVEL, addr(level)),
                (C_MAXLVL, addr(maxlvl)), (C_NBCNT, addr(nbcnt)),
                (C_DU, addr(du)), (C_DW, addr(dw)),
                (C_STAMPU, addr(stampu)), (C_STAMPW, addr(stampw)),
                (C_ORDER, addr(order)), (C_OCOUNT, addr(ocount)),
                (C_OHK, addr(ohk)), (C_OHV, addr(ohv)), (C_OHS, addr(ohs)),
                (C_WHK, addr(whk)), (C_WHV, addr(whv)), (C_WHS, addr(whs)),
            ])
            made = lib.gh_ch_prepare(n, ctx)
            if made >= 0:
                break
            cap = 2 * cap
            if cap > 64 * (m + n) + 1024:
                raise RuntimeError("contraction hierarchy did not fit its buffers")

        self.shortcuts = int(made)
        self.base_entries = int(used[2])
        self.levels = level
        self._fhead, self._fnxt, self._fto, self._fw = fhead, fnxt, fto, fw
        self._fmid = fmid
        self._rmid = rmid
        self._rhead, self._rnxt, self._rto, self._rw = rhead, rnxt, rto, rw
        self._df, self._dt, self._sf, self._st = df, dt, sf, st
        self._dist, self._prev_edge, self._done = dist, prev, done
        self._prev_node = prev_node
        self.cost = 0.0
        self._hk, self._hv, self._hs = hk, hv, hs
        self._mid_args = Args()
        self._down_args = Args()

    def _star(self, reverse: bool):
        if reverse:
            return self._rhead, self._rnxt, self._rto, self._rw, self._rmid
        return self._fhead, self._fnxt, self._fto, self._fw, self._fmid

    def _star_edge(self, head, nxt, to, w, u: int, v: int) -> int:
        """Cheapest star edge u->v, or -1."""
        best = -1
        best_w = BIG
        e = int(head[u])
        while e >= 0:
            if int(to[e]) == v and w[e] < best_w:
                best_w = w[e]
                best = e
            e = int(nxt[e])
        return best

    def _unpack(self, start: int, root: int, reverse: bool = False) -> np.ndarray:
        """Expand a CH path into a base-graph node path, from `start` to `root`.

        The downward search records, for every node, the node one hop closer to
        the root, which gives the CH path hop by hop; walking it from the root
        outwards means every hop is a genuine edge of the star being unpacked,
        whichever of the two mirrored stars that is. Each hop is then expanded:
        a star edge u->v carrying a middle node stands for u->m->v, and both
        halves may themselves be shortcuts, so the expansion recurses. It uses
        an explicit work list rather than Python recursion because a long route
        through a big hierarchy nests deeply enough to matter.
        """
        chain = [int(start)]
        cur = int(start)
        while cur != root:
            cur = int(self._prev_node[cur])
            if cur < 0:
                raise RuntimeError(f"no CH chain from {start} back to {root}")
            chain.append(cur)
            if len(chain) > self.graph.n:
                raise RuntimeError("CH predecessor chain is cyclic")
        chain.reverse()
        head, nxt, to, w, mid = self._star(reverse)
        out = []
        budget = self.graph.n + self.shortcuts + 4
        for a, b in zip(chain[:-1], chain[1:]):
            work = [(a, b)]
            while work:
                u, target = work.pop()
                budget -= 1
                if budget < 0:
                    raise RuntimeError("CH shortcut expansion did not terminate")
                if u == target:
                    continue
                e = self._star_edge(head, nxt, to, w, u, target)
                if e < 0:
                    raise RuntimeError(
                        f"no CH edge {u}->{target} while unpacking"
                    )
                m = int(mid[e])
                if m < 0:
                    out.append(u)
                else:
                    work.append((m, target))
                    work.append((u, m))
        out.append(int(start))
        out.reverse()
        return np.asarray(out, dtype=np.int64)

    def route(self, src: int, dst: int) -> Optional[Path]:
        """Shortest path through the hierarchy. None when unreachable."""
        g = self.graph
        g._check_nodes(src, dst)
        mid_ctx = self._mid_args.set([
            (C_FHEAD, addr(self._fhead)), (C_FNXT, addr(self._fnxt)),
            (C_FTO, addr(self._fto)), (C_FW, addr(self._fw)),
            (C_RHEAD, addr(self._rhead)), (C_RNXT, addr(self._rnxt)),
            (C_RTO, addr(self._rto)), (C_RW, addr(self._rw)),
            (C_LEVEL, addr(self.levels)),
            (M_DF, addr(self._df)), (M_DT, addr(self._dt)),
            (M_SF, addr(self._sf)), (M_ST, addr(self._st)),
            (M_HK, addr(self._hk)), (M_HV, addr(self._hv)),
            (M_HS, addr(self._hs)),
        ])
        middle = lib.gh_ch_middle(g.n, int(src), int(dst), mid_ctx)
        if middle < 0:
            return None
        self.cost = float(self._df[middle]) + float(self._dt[middle])
        down_ctx = self._down_args.set([
            (C_FHEAD, addr(self._fhead)), (C_FNXT, addr(self._fnxt)),
            (C_FTO, addr(self._fto)), (C_FW, addr(self._fw)),
            (C_FMID, addr(self._fmid)),
            (C_RHEAD, addr(self._rhead)), (C_RNXT, addr(self._rnxt)),
            (C_RTO, addr(self._rto)), (C_RW, addr(self._rw)),
            (C_RMID, addr(self._rmid)),
            (C_LEVEL, addr(self.levels)),
            (S_DIST, addr(self._dist)), (S_PREV, addr(self._prev_edge)),
            (S_PREVN, addr(self._prev_node)), (S_DONE, addr(self._done)),
            (S_HK, addr(self._hk)), (S_HV, addr(self._hv)),
            (S_HS, addr(self._hs)),
        ])
        # The source-to-middle half is retraced over the reverse star (those
        # are the backwards of the forward edges the upward search used) and the
        # middle-to-target half over the forward star.
        lib.gh_ch_downward(g.n, int(middle), 1, down_ctx)
        if self._dist[src] >= BIG:
            return None
        left = self._unpack(int(src), int(middle), reverse=True)
        lib.gh_ch_downward(g.n, int(middle), 0, down_ctx)
        if self._dist[dst] >= BIG:
            return None
        right = self._unpack(int(dst), int(middle))
        nodes = np.concatenate([left, right[::-1][1:]])
        return g._finish(nodes)

    def star_entries(self) -> int:
        """Forward-star entry count, a shortcut-density diagnostic."""
        total = 0
        for v in range(self.graph.n):
            e = int(self._fhead[v])
            while e >= 0:
                total += 1
                e = int(self._fnxt[e])
        return total


def grid_graph(rows: int, cols: int, spacing_deg: float = 0.01,
               speed: float = 13.89, directed: bool = False,
               one_way_stride: int = 0, slow_column_stride: int = 11) -> Graph:
    """A lat/lon grid, the test stand-in for a road network.

    Unit weights are useless to a shortest-path test, so every
    `slow_column_stride`-th column is half speed, and when `one_way_stride` is
    positive every n-th column pair carries traffic in one direction only.
    `one_way_stride <= 0` means a fully two-way grid.
    """
    n = rows * cols
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            lat[r * cols + c] = 52.0 + r * spacing_deg
            lon[r * cols + c] = 13.0 + c * spacing_deg
    edges = []
    speeds = []

    def node(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            sp = speed * (0.5 if c % slow_column_stride == 0 else 1.0)
            if c + 1 < cols:
                one_way = one_way_stride > 0 and c % one_way_stride == 0
                blocked_back = one_way_stride > 0 and (
                    c % one_way_stride == one_way_stride - 1
                )
                if not one_way:
                    edges.append((node(r, c), node(r, c + 1)))
                    speeds.append(sp)
                if directed or not blocked_back:
                    edges.append((node(r, c + 1), node(r, c)))
                    speeds.append(sp)
            if r + 1 < rows:
                edges.append((node(r, c), node(r + 1, c)))
                speeds.append(sp)
                edges.append((node(r + 1, c), node(r, c)))
                speeds.append(sp)
    return Graph(lat, lon, np.asarray(edges, dtype=np.int64),
                 np.asarray(speeds, dtype=np.float64), directed=directed)


def reference_shortest_path(graph: Graph, src: int, dst: int):
    """Independent shortest path by exhaustive Bellman-Ford, for cross-checks.

    Deliberately not the same algorithm as the Mojo kernels: O(V*E), no heap, no
    levels and no shortcuts, so an error in the heap, the level logic or the
    witness search cannot hide inside it. Returns (nodes, weight) or
    (None, inf) when unreachable.
    """
    n = graph.n
    head, to, w = graph._head, graph._to, graph._weight
    dist = [math.inf] * n
    dist[src] = 0.0
    for _ in range(n):
        changed = False
        for u in range(n):
            if dist[u] == math.inf:
                continue
            for e in range(int(head[u]), int(head[u + 1])):
                v = int(to[e])
                nd = dist[u] + float(w[e])
                if nd < dist[v]:
                    dist[v] = nd
                    changed = True
        if not changed:
            break
    if dist[dst] == math.inf:
        return None, math.inf
    nodes = [dst]
    cur = dst
    for _ in range(n + 1):
        best = None
        bestd = dist[cur]
        for e in range(int(head[cur]), int(head[cur + 1])):
            v = int(to[e])
            cand = dist[v] + float(w[e])
            if cand < bestd - 1e-12:
                bestd = cand
                best = v
        if best is None:
            break
        cur = best
        nodes.append(cur)
    return nodes[::-1], dist[dst]
