"""Routing kernels for the GraphHopper algorithm port.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric. All scratch space
is allocated and owned by the Python shim; nothing here allocates.

Graphs cross the ABI as CSR (head/target/weight). The contraction-hierarchy
preprocessing keeps a forward-star adjacency (forward plus reverse) because
shortcut insertion has to grow both directions, and CH queries are the only
consumer of that structure.

ABI NOTE -- read before adding a parameter to any export below.

On this toolchain (mojo 1.2.0.dev2026092605) an `@export ... abi("C")` function
with seven or more parameters miscompiles: integer comparisons on arguments the
SysV ABI passes on the stack are evaluated wrongly. A probe over parameter
counts 6..20 found the six register-passed arguments reliable and every count
from 7 upwards intermittently evaluating a two-term `or` of comparisons as true,
nondeterministically and differently in two files with identical bodies.

So no export here has more than four parameters. Scalars that are compared (n,
src, dst, middle) are passed directly; every buffer address lives in a
caller-owned Int64 "argument block" whose address is itself one of the four.
This is not a stylistic choice: an earlier revision of this file passed
gh_astar (n, head, to, w, heur, src, dst, ...) with dst in the seventh slot and
every query came back reporting a bad node index.
"""

from std.math import asin, atan2, cos, sin, sqrt

comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

# DBL_MAX. Every reachable weight is a finite sum of non-negative edge weights
# far below this, and every "unreached" slot in a distance array starts here, so
# `new < dist[v]` is false until a real path finds the node. Using the largest
# finite double instead of a true infinity keeps every comparison total and
# avoids any NaN interaction in the heap.
comptime BIG: Float64 = 1.7976931348623157e308

comptime EARTH_RADIUS_M: Float64 = 6371000.0

comptime DEG2RAD: Float64 = 3.141592653589793 / 180.0
comptime RAD2DEG: Float64 = 180.0 / 3.141592653589793


def f64p(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


# Argument-block slot numbers, mirrored by mojo_graphhopperpy._lib.
#
# The four groups sit in disjoint ranges on purpose. An earlier revision packed
# them all against zero, where S_HEAD and C_FHEAD were both slot 0; a CH prepare
# filled the block with both and the second write won, so the kernel read the
# forward-star head where it expected the CSR head and ran off the end of it.
comptime G_BASE: Int = 0  # geometry: 6 slots
comptime S_BASE: Int = 8  # searches: 10 slots
comptime C_BASE: Int = 32  # contraction hierarchy: 26 slots
comptime M_BASE: Int = 64  # CH middle-node search: 7 slots
comptime CTX_SLOTS: Int = 72

comptime G_LAT_A: Int = G_BASE + 0
comptime G_LON_A: Int = G_BASE + 1
comptime G_LAT_B: Int = G_BASE + 2
comptime G_LON_B: Int = G_BASE + 3
comptime G_OUT: Int = G_BASE + 4
comptime G_SPEED: Int = G_BASE + 5

comptime S_HEAD: Int = S_BASE + 0
comptime S_TO: Int = S_BASE + 1
comptime S_W: Int = S_BASE + 2
comptime S_DIST: Int = S_BASE + 3
comptime S_PREV: Int = S_BASE + 4
comptime S_DONE: Int = S_BASE + 5
comptime S_HK: Int = S_BASE + 6
comptime S_HV: Int = S_BASE + 7
comptime S_HS: Int = S_BASE + 8
comptime S_HEUR: Int = S_BASE + 9
comptime S_PREVN: Int = S_BASE + 10

comptime C_FHEAD: Int = C_BASE + 0
comptime C_FNXT: Int = C_BASE + 1
comptime C_FTO: Int = C_BASE + 2
comptime C_FW: Int = C_BASE + 3
comptime C_FMID: Int = C_BASE + 4
comptime C_RHEAD: Int = C_BASE + 5
comptime C_RNXT: Int = C_BASE + 6
comptime C_RTO: Int = C_BASE + 7
comptime C_RW: Int = C_BASE + 8
comptime C_RMID: Int = C_BASE + 9
comptime C_USED: Int = C_BASE + 10
comptime C_LEVEL: Int = C_BASE + 11
comptime C_MAXLVL: Int = C_BASE + 12
comptime C_NBCNT: Int = C_BASE + 13
comptime C_DU: Int = C_BASE + 14
comptime C_DW: Int = C_BASE + 15
comptime C_STAMPU: Int = C_BASE + 16
comptime C_STAMPW: Int = C_BASE + 17
comptime C_ORDER: Int = C_BASE + 18
comptime C_OCOUNT: Int = C_BASE + 19
comptime C_OHK: Int = C_BASE + 20
comptime C_OHV: Int = C_BASE + 21
comptime C_OHS: Int = C_BASE + 22
comptime C_WHK: Int = C_BASE + 23
comptime C_WHV: Int = C_BASE + 24
comptime C_WHS: Int = C_BASE + 25

comptime M_DF: Int = M_BASE + 0
comptime M_DT: Int = M_BASE + 1
comptime M_SF: Int = M_BASE + 2
comptime M_ST: Int = M_BASE + 3
comptime M_HK: Int = M_BASE + 4
comptime M_HV: Int = M_BASE + 5
comptime M_HS: Int = M_BASE + 6


# ---------------------------------------------------------------------------
# Great-circle geometry. GraphHopper's DistanceEarthProjection is haversine on a
# sphere of radius 6371000 m; DistancePlaneProjection is the equirectangular
# approximation, which is what it falls back to for short hops.
# ---------------------------------------------------------------------------


@export("gh_haversine")
def gh_haversine(n: Int, ctx_addr: Int) abi("C"):
    """Great-circle metres between point A i and point B i."""
    var ctx = i64p(ctx_addr)
    var lat_a = f64p(Int(ctx[unsafe_offset=G_LAT_A]))
    var lon_a = f64p(Int(ctx[unsafe_offset=G_LON_A]))
    var lat_b = f64p(Int(ctx[unsafe_offset=G_LAT_B]))
    var lon_b = f64p(Int(ctx[unsafe_offset=G_LON_B]))
    var res = f64p(Int(ctx[unsafe_offset=G_OUT]))
    for i in range(n):
        var p1 = lat_a[unsafe_offset=i] * DEG2RAD
        var p2 = lat_b[unsafe_offset=i] * DEG2RAD
        var dp = (lat_b[unsafe_offset=i] - lat_a[unsafe_offset=i]) * DEG2RAD
        var dl = (lon_b[unsafe_offset=i] - lon_a[unsafe_offset=i]) * DEG2RAD
        var s1 = sin(dp * 0.5)
        var s2 = sin(dl * 0.5)
        var a = s1 * s1 + cos(p1) * cos(p2) * s2 * s2
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        res[unsafe_offset=i] = 2.0 * EARTH_RADIUS_M * asin(sqrt(a))


@export("gh_equirect")
def gh_equirect(n: Int, ctx_addr: Int) abi("C"):
    """Equirectangular metres metres between point A i and point B i.

    GraphHopper's DistancePlaneProjection: project both points onto a plane at
    the mean latitude, with the longitude difference scaled by cos(mean_lat).
    """
    var ctx = i64p(ctx_addr)
    var lat_a = f64p(Int(ctx[unsafe_offset=G_LAT_A]))
    var lon_a = f64p(Int(ctx[unsafe_offset=G_LON_A]))
    var lat_b = f64p(Int(ctx[unsafe_offset=G_LAT_B]))
    var lon_b = f64p(Int(ctx[unsafe_offset=G_LON_B]))
    var res = f64p(Int(ctx[unsafe_offset=G_OUT]))
    for i in range(n):
        var dlat = (lat_b[unsafe_offset=i] - lat_a[unsafe_offset=i]) * DEG2RAD
        var dlon = (lon_b[unsafe_offset=i] - lon_a[unsafe_offset=i]) * DEG2RAD
        var mean_lat = (lat_a[unsafe_offset=i] + lat_b[unsafe_offset=i]) * 0.5
        var x = dlon * cos(mean_lat * DEG2RAD)
        res[unsafe_offset=i] = EARTH_RADIUS_M * sqrt(x * x + dlat * dlat)


@export("gh_bearing")
def gh_bearing(n: Int, ctx_addr: Int) abi("C"):
    """Initial bearing in degrees in degrees from point A i to point B i."""
    var ctx = i64p(ctx_addr)
    var lat_a = f64p(Int(ctx[unsafe_offset=G_LAT_A]))
    var lon_a = f64p(Int(ctx[unsafe_offset=G_LON_A]))
    var lat_b = f64p(Int(ctx[unsafe_offset=G_LAT_B]))
    var lon_b = f64p(Int(ctx[unsafe_offset=G_LON_B]))
    var res = f64p(Int(ctx[unsafe_offset=G_OUT]))
    for i in range(n):
        var p1 = lat_a[unsafe_offset=i] * DEG2RAD
        var p2 = lat_b[unsafe_offset=i] * DEG2RAD
        var dl = (lon_b[unsafe_offset=i] - lon_a[unsafe_offset=i]) * DEG2RAD
        var y = sin(dl) * cos(p2)
        var x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
        var brg = atan2(y, x) * RAD2DEG
        if brg < 0.0:
            brg = brg + 360.0
        res[unsafe_offset=i] = brg


@export("gh_edge_weights")
def gh_edge_weights(n: Int, ctx_addr: Int) abi("C"):
    """Edge weight: haversine metres / speed[i].

    GraphHopper's DistanceWeighting followed by a speed conversion: the routing
    weight of an edge is its length divided by the speed on it. A non-positive
    speed yields BIG, which the Python layer rejects before it gets here.
    """
    var ctx = i64p(ctx_addr)
    var lat_a = f64p(Int(ctx[unsafe_offset=G_LAT_A]))
    var lon_a = f64p(Int(ctx[unsafe_offset=G_LON_A]))
    var lat_b = f64p(Int(ctx[unsafe_offset=G_LAT_B]))
    var lon_b = f64p(Int(ctx[unsafe_offset=G_LON_B]))
    var speed = f64p(Int(ctx[unsafe_offset=G_SPEED]))
    var res = f64p(Int(ctx[unsafe_offset=G_OUT]))
    for i in range(n):
        var p1 = lat_a[unsafe_offset=i] * DEG2RAD
        var p2 = lat_b[unsafe_offset=i] * DEG2RAD
        var dp = (lat_b[unsafe_offset=i] - lat_a[unsafe_offset=i]) * DEG2RAD
        var dl = (lon_b[unsafe_offset=i] - lon_a[unsafe_offset=i]) * DEG2RAD
        var s1 = sin(dp * 0.5)
        var s2 = sin(dl * 0.5)
        var a = s1 * s1 + cos(p1) * cos(p2) * s2 * s2
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0
        var dist = 2.0 * EARTH_RADIUS_M * asin(sqrt(a))
        var s = speed[unsafe_offset=i]
        if s <= 0.0:
            res[unsafe_offset=i] = BIG
        else:
            res[unsafe_offset=i] = dist / s


# ---------------------------------------------------------------------------
# Binary min-heap over parallel (key, value) arrays, owned by the caller.
# ---------------------------------------------------------------------------


def heap_swap(k: F64Ptr, v: I32Ptr, i: Int, j: Int):
    var tk = k[unsafe_offset=i]
    k[unsafe_offset=i] = k[unsafe_offset=j]
    k[unsafe_offset=j] = tk
    var tv = v[unsafe_offset=i]
    v[unsafe_offset=i] = v[unsafe_offset=j]
    v[unsafe_offset=j] = tv


def heap_push(k: F64Ptr, v: I32Ptr, size_addr: I64Ptr, key: Float64, val: Int32):
    var s = Int(size_addr[unsafe_offset=0])
    var i = s
    k[unsafe_offset=i] = key
    v[unsafe_offset=i] = val
    while i > 0:
        var par = (i - 1) >> 1
        if k[unsafe_offset=par] <= k[unsafe_offset=i]:
            break
        heap_swap(k, v, i, par)
        i = par
    size_addr[unsafe_offset=0] = Int64(s + 1)


def heap_pop(k: F64Ptr, v: I32Ptr, size_addr: I64Ptr) -> Tuple[Float64, Int32]:
    var s = Int(size_addr[unsafe_offset=0])
    if s == 0:
        return (BIG, Int32(-1))
    var top_k = k[unsafe_offset=0]
    var top_v = v[unsafe_offset=0]
    s -= 1
    size_addr[unsafe_offset=0] = Int64(s)
    if s == 0:
        return (top_k, top_v)
    k[unsafe_offset=0] = k[unsafe_offset=s]
    v[unsafe_offset=0] = v[unsafe_offset=s]
    var i = 0
    while True:
        var left = 2 * i + 1
        if left >= s:
            break
        var right = left + 1
        var m = i
        if k[unsafe_offset=left] < k[unsafe_offset=m]:
            m = left
        if right < s and k[unsafe_offset=right] < k[unsafe_offset=m]:
            m = right
        if m == i:
            break
        heap_swap(k, v, i, m)
        i = m
    return (top_k, top_v)


def reset_dist(n: Int, dist: F64Ptr, prev: I32Ptr, done: U8Ptr):
    for i in range(n):
        dist[unsafe_offset=i] = BIG
        prev[unsafe_offset=i] = Int32(-1)
        done[unsafe_offset=i] = UInt8(0)


@export("gh_dijkstra")
def gh_dijkstra(n: Int, src: Int, dst: Int, ctx_addr: Int) abi("C") -> Int:
    """Shortest path by edge weight. 1 found, 0 unreachable, -1 bad node.

    The loop is written out here rather than shared with gh_astar so that each
    entry point owns its argument list end to end.
    """
    var ctx = i64p(ctx_addr)
    var head = i64p(Int(ctx[unsafe_offset=S_HEAD]))
    var to = i32p(Int(ctx[unsafe_offset=S_TO]))
    var w = f64p(Int(ctx[unsafe_offset=S_W]))
    var dist = f64p(Int(ctx[unsafe_offset=S_DIST]))
    var prev = i32p(Int(ctx[unsafe_offset=S_PREV]))
    var done = u8p(Int(ctx[unsafe_offset=S_DONE]))
    var hk = f64p(Int(ctx[unsafe_offset=S_HK]))
    var hv = i32p(Int(ctx[unsafe_offset=S_HV]))
    var hs = i64p(Int(ctx[unsafe_offset=S_HS]))
    reset_dist(n, dist, prev, done)
    hs[unsafe_offset=0] = Int64(0)
    if src < 0 or src >= n or dst < 0 or dst >= n:
        return -1
    if src == dst:
        return 1
    dist[unsafe_offset=src] = Float64(0.0)
    heap_push(hk, hv, hs, Float64(0.0), Int32(src))
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = Int(item[1])
        if done[unsafe_offset=u] == UInt8(1):
            continue
        done[unsafe_offset=u] = UInt8(1)
        if u == dst:
            return 1
        var g = dist[unsafe_offset=u]
        var e = Int(head[unsafe_offset=u])
        var stop = Int(head[unsafe_offset=u + 1])
        while e < stop:
            var v = Int(to[unsafe_offset=e])
            if done[unsafe_offset=v] == UInt8(0):
                var ng = g + w[unsafe_offset=e]
                if ng < dist[unsafe_offset=v]:
                    dist[unsafe_offset=v] = ng
                    prev[unsafe_offset=v] = Int32(u)
                    heap_push(hk, hv, hs, ng, Int32(v))
            e += 1
    return 0


@export("gh_astar")
def gh_astar(n: Int, src: Int, dst: Int, ctx_addr: Int) abi("C") -> Int:
    """A* with a precomputed per-node heuristic. Same codes as gh_dijkstra.

    The heuristic must be admissible, i.e. never larger than the true remaining
    cost, or the target is settled too early and the answer is wrong. It does
    not have to be consistent for correctness, only for the node-expansion
    count; the Python layer uses a great-circle lower bound, which is both.
    """
    var ctx = i64p(ctx_addr)
    var head = i64p(Int(ctx[unsafe_offset=S_HEAD]))
    var to = i32p(Int(ctx[unsafe_offset=S_TO]))
    var w = f64p(Int(ctx[unsafe_offset=S_W]))
    var heur = f64p(Int(ctx[unsafe_offset=S_HEUR]))
    var dist = f64p(Int(ctx[unsafe_offset=S_DIST]))
    var prev = i32p(Int(ctx[unsafe_offset=S_PREV]))
    var done = u8p(Int(ctx[unsafe_offset=S_DONE]))
    var hk = f64p(Int(ctx[unsafe_offset=S_HK]))
    var hv = i32p(Int(ctx[unsafe_offset=S_HV]))
    var hs = i64p(Int(ctx[unsafe_offset=S_HS]))
    reset_dist(n, dist, prev, done)
    hs[unsafe_offset=0] = Int64(0)
    if src < 0 or src >= n or dst < 0 or dst >= n:
        return -1
    if src == dst:
        return 1
    dist[unsafe_offset=src] = Float64(0.0)
    heap_push(hk, hv, hs, heur[unsafe_offset=src], Int32(src))
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = Int(item[1])
        if done[unsafe_offset=u] == UInt8(1):
            continue
        done[unsafe_offset=u] = UInt8(1)
        if u == dst:
            return 1
        var g = dist[unsafe_offset=u]
        var e = Int(head[unsafe_offset=u])
        var stop = Int(head[unsafe_offset=u + 1])
        while e < stop:
            var v = Int(to[unsafe_offset=e])
            if done[unsafe_offset=v] == UInt8(0):
                var ng = g + w[unsafe_offset=e]
                if ng < dist[unsafe_offset=v]:
                    dist[unsafe_offset=v] = ng
                    prev[unsafe_offset=v] = Int32(u)
                    heap_push(hk, hv, hs, ng + heur[unsafe_offset=v], Int32(v))
            e += 1
    return 0


# ---------------------------------------------------------------------------
# Contraction hierarchies.
#
# Forward-star adjacency, because shortcuts are appended while contracting. A
# node's level is 0 while it is uncontracted and 1 + (deepest already contracted
# neighbour) once contracted, so level 1 is the first node removed and level 0
# is a node that is never removed. An edge v->u is an *upward* edge exactly when
# level[u] > level[v]: upward means toward the nodes that survive longest. The
# upward search terminates because a contracted node's non-shortcut edges all
# lead back down.
# ---------------------------------------------------------------------------


def fs_find(
    head: I32Ptr, nxt: I32Ptr, to: I32Ptr, w: F64Ptr, src: Int32, dst: Int32
) -> Float64:
    """Cheapest weight of an edge src->dst in a forward star, or BIG if absent."""
    var e = Int(head[unsafe_offset=src])
    while e >= 0:
        if to[unsafe_offset=e] == dst:
            return w[unsafe_offset=e]
        e = Int(nxt[unsafe_offset=e])
    return BIG


def fs_add(
    head: I32Ptr,
    nxt: I32Ptr,
    to: I32Ptr,
    w: F64Ptr,
    mid: I32Ptr,
    used_addr: I32Ptr,
    src: Int32,
    dst: Int32,
    weight: Float64,
    middle: Int32,
) -> Bool:
    """Push an edge onto the front of src's list. False when out of capacity."""
    var slot = Int(used_addr[unsafe_offset=0])
    if slot >= Int(used_addr[unsafe_offset=1]):
        return False
    to[unsafe_offset=slot] = dst
    w[unsafe_offset=slot] = weight
    mid[unsafe_offset=slot] = middle
    nxt[unsafe_offset=slot] = head[unsafe_offset=src]
    head[unsafe_offset=src] = Int32(slot)
    used_addr[unsafe_offset=0] = Int32(slot + 1)
    return True


def collect_nbrs(
    head: I32Ptr, nxt: I32Ptr, to: I32Ptr, level: I32Ptr, v: Int32, buf: I32Ptr
) -> Int:
    """Uncontracted out-neighbours of v, de-duplicated, into `buf`."""
    var count = 0
    var e = Int(head[unsafe_offset=v])
    while e >= 0:
        var u = to[unsafe_offset=e]
        if level[unsafe_offset=u] == 0:
            var seen = False
            var k = 0
            while k < count:
                if buf[unsafe_offset=k] == u:
                    seen = True
                    break
                k += 1
            if not seen:
                buf[unsafe_offset=count] = u
                count += 1
        e = Int(nxt[unsafe_offset=e])
    return count


def bounded_search(
    head: I32Ptr,
    nxt: I32Ptr,
    to: I32Ptr,
    w: F64Ptr,
    level: I32Ptr,
    src: Int32,
    limit: Float64,
    d: F64Ptr,
    stamp: I32Ptr,
    tag: Int32,
    order: I32Ptr,
    ocount_addr: I32Ptr,
    hk: F64Ptr,
    hv: I32Ptr,
    hs: I64Ptr,
) -> Int:
    """Bounded Dijkstra over uncontracted nodes only; records the popped order.

    `stamp` avoids clearing `d` between searches: a relaxation is only allowed
    when the target was not reached by this search at all, or was reached at a
    larger distance. Without that guard a stale small value from an earlier
    search silently blocks the relaxation and the witness is missed.
    """
    hs[unsafe_offset=0] = Int64(0)
    var count = 0
    d[unsafe_offset=src] = Float64(0.0)
    stamp[unsafe_offset=src] = tag
    heap_push(hk, hv, hs, Float64(0.0), src)
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = item[1]
        var du = item[0]
        if du > d[unsafe_offset=u]:
            continue
        order[unsafe_offset=count] = u
        count += 1
        if du >= limit:
            break
        var e = Int(head[unsafe_offset=u])
        while e >= 0:
            var v = to[unsafe_offset=e]
            if level[unsafe_offset=v] == 0:
                var nd = du + w[unsafe_offset=e]
                if nd < limit:
                    if stamp[unsafe_offset=v] != tag:
                        stamp[unsafe_offset=v] = tag
                        d[unsafe_offset=v] = nd
                        heap_push(hk, hv, hs, nd, v)
                    elif nd < d[unsafe_offset=v]:
                        d[unsafe_offset=v] = nd
                        heap_push(hk, hv, hs, nd, v)
            e = Int(nxt[unsafe_offset=e])
    ocount_addr[unsafe_offset=0] = Int32(count)
    return count


def _contract(
    n: Int,
    ctx: I64Ptr,
) -> Int:
    """Greedy contraction with witness search. See gh_ch_prepare for the codes.

    The context block is re-read inside the loop instead of being hoisted into
    locals. Twenty-six hoisted pointers is far more live state than the register
    file holds, and the version of this function that did so was miscompiled on
    this toolchain into a wild store.
    """
    var head = i64p(Int(ctx[unsafe_offset=S_HEAD]))
    var to = i32p(Int(ctx[unsafe_offset=S_TO]))
    var w = f64p(Int(ctx[unsafe_offset=S_W]))
    var used = i32p(Int(ctx[unsafe_offset=C_USED]))

    used[unsafe_offset=0] = Int32(0)
    used[unsafe_offset=2] = Int32(0)

    for v in range(n):
        i32p(Int(ctx[unsafe_offset=C_FHEAD]))[unsafe_offset=v] = Int32(-1)
        i32p(Int(ctx[unsafe_offset=C_RHEAD]))[unsafe_offset=v] = Int32(-1)
        i32p(Int(ctx[unsafe_offset=C_LEVEL]))[unsafe_offset=v] = Int32(0)
        i32p(Int(ctx[unsafe_offset=C_MAXLVL]))[unsafe_offset=v] = Int32(0)
        i32p(Int(ctx[unsafe_offset=C_NBCNT]))[unsafe_offset=v] = Int32(0)
    for v in range(n):
        var e = Int(head[unsafe_offset=v])
        var stop = Int(head[unsafe_offset=v + 1])
        while e < stop:
            var u = to[unsafe_offset=e]
            var weight = w[unsafe_offset=e]
            var ok = fs_add(
                i32p(Int(ctx[unsafe_offset=C_FHEAD])),
                i32p(Int(ctx[unsafe_offset=C_FNXT])),
                i32p(Int(ctx[unsafe_offset=C_FTO])),
                f64p(Int(ctx[unsafe_offset=C_FW])),
                i32p(Int(ctx[unsafe_offset=C_FMID])),
                used, Int32(v), Int32(u), weight, Int32(-1),
            )
            if not ok:
                used[unsafe_offset=2] = used[unsafe_offset=0]
                return -2
            ok = fs_add(
                i32p(Int(ctx[unsafe_offset=C_RHEAD])),
                i32p(Int(ctx[unsafe_offset=C_RNXT])),
                i32p(Int(ctx[unsafe_offset=C_RTO])),
                f64p(Int(ctx[unsafe_offset=C_RW])),
                i32p(Int(ctx[unsafe_offset=C_RMID])),
                used, Int32(u), Int32(v), weight, Int32(-1),
            )
            if not ok:
                used[unsafe_offset=2] = used[unsafe_offset=0]
                return -2
            e += 1
    used[unsafe_offset=2] = used[unsafe_offset=0]

    # Contraction order: a lazy max-heap on a greedy priority. The priority
    # affects speed, never correctness, but a bad order produces an unusable
    # hierarchy, so it is recomputed as neighbours get contracted. The witness
    # searches below need their own heap: this one has live entries across them.
    var ohk = f64p(Int(ctx[unsafe_offset=C_OHK]))
    var ohv = i32p(Int(ctx[unsafe_offset=C_OHV]))
    var ohs = i64p(Int(ctx[unsafe_offset=C_OHS]))
    ohs[unsafe_offset=0] = Int64(0)
    for v in range(n):
        var deg = Int32(0)
        var e = Int(i32p(Int(ctx[unsafe_offset=C_FHEAD]))[unsafe_offset=v])
        while e >= 0:
            deg += Int32(1)
            e = Int(i32p(Int(ctx[unsafe_offset=C_FNXT]))[unsafe_offset=e])
        heap_push(ohk, ohv, ohs, -Float64(deg), Int32(v))

    var shortcuts = 0
    var tag = Int32(1)
    while Int(ohs[unsafe_offset=0]) > 0:
        var item = heap_pop(ohk, ohv, ohs)
        var v = item[1]
        var level = i32p(Int(ctx[unsafe_offset=C_LEVEL]))
        if level[unsafe_offset=v] != 0:
            continue

        # Contract v: its level is one deeper than its deepest contracted
        # neighbour, which is what keeps every remaining edge upward-valid.
        var maxlvl = i32p(Int(ctx[unsafe_offset=C_MAXLVL]))
        var lv = maxlvl[unsafe_offset=v] + 1
        level[unsafe_offset=v] = lv
        var fhead = i32p(Int(ctx[unsafe_offset=C_FHEAD]))
        var fnxt = i32p(Int(ctx[unsafe_offset=C_FNXT]))
        var fto = i32p(Int(ctx[unsafe_offset=C_FTO]))
        var nbcnt = i32p(Int(ctx[unsafe_offset=C_NBCNT]))
        var e = Int(fhead[unsafe_offset=v])
        while e >= 0:
            var u = fto[unsafe_offset=e]
            if level[unsafe_offset=u] == 0:
                if lv > maxlvl[unsafe_offset=u]:
                    maxlvl[unsafe_offset=u] = lv
                var c = nbcnt[unsafe_offset=u] + 1
                nbcnt[unsafe_offset=u] = c
                var dg = Int32(0)
                var e2 = Int(fhead[unsafe_offset=u])
                while e2 >= 0:
                    dg += Int32(1)
                    e2 = Int(fnxt[unsafe_offset=e2])
                heap_push(
                    ohk, ohv, ohs, -Float64(dg) - 2.0 * Float64(c), Int32(u)
                )
            e = Int(fnxt[unsafe_offset=e])

        # out-neighbours at [0, n), in-neighbours at [n, 2n), and the two
        # witness pop-lists at [2n, 3n) and [3n, 4n). They must not overlap:
        # the witness searches run while the neighbour lists are still read.
        var order = i32p(Int(ctx[unsafe_offset=C_ORDER]))
        var fw = f64p(Int(ctx[unsafe_offset=C_FW]))
        var outs = collect_nbrs(
            fhead, fnxt, fto, level, v, order
        )
        if outs <= 1:
            continue
        var ins_buf = order.unsafe_offset(n)
        var ins = collect_nbrs(
            i32p(Int(ctx[unsafe_offset=C_RHEAD])),
            i32p(Int(ctx[unsafe_offset=C_RNXT])),
            i32p(Int(ctx[unsafe_offset=C_RTO])),
            level, v, ins_buf,
        )
        if ins <= 1:
            continue

        var ocount = i32p(Int(ctx[unsafe_offset=C_OCOUNT]))
        var du = f64p(Int(ctx[unsafe_offset=C_DU]))
        var dw = f64p(Int(ctx[unsafe_offset=C_DW]))
        var stampu = i32p(Int(ctx[unsafe_offset=C_STAMPU]))
        var stampw = i32p(Int(ctx[unsafe_offset=C_STAMPW]))
        var whk = f64p(Int(ctx[unsafe_offset=C_WHK]))
        var whv = i32p(Int(ctx[unsafe_offset=C_WHV]))
        var whs = i64p(Int(ctx[unsafe_offset=C_WHS]))
        var a = 0
        while a < ins:
            var uu = order[unsafe_offset=n + a]
            var wuv = fs_find(fhead, fnxt, fto, fw, uu, v)
            if wuv >= BIG:
                a += 1
                continue
            var b = 0
            while b < outs:
                var ww = order[unsafe_offset=b]
                if ww == uu:
                    # An in-neighbour and an out-neighbour can be the same node
                    # whenever the graph has edges in both directions between it
                    # and v, which is every neighbour of an undirected graph.
                    # Contracting that "pair" would add a self-loop shortcut.
                    b += 1
                    continue
                var wvw = fs_find(fhead, fnxt, fto, fw, v, ww)
                if wvw >= BIG:
                    b += 1
                    continue
                var shortcut = wuv + wvw
                tag += 1
                var fwd_buf = order.unsafe_offset(2 * n)
                _ = bounded_search(
                    fhead, fnxt, fto, fw, level, uu, shortcut, du, stampu,
                    tag, fwd_buf, ocount, whk, whv, whs,
                )
                var fwd_n = Int(ocount[unsafe_offset=0])
                tag += 1
                var rev_buf = order.unsafe_offset(3 * n)
                _ = bounded_search(
                    i32p(Int(ctx[unsafe_offset=C_RHEAD])),
                    i32p(Int(ctx[unsafe_offset=C_RNXT])),
                    i32p(Int(ctx[unsafe_offset=C_RTO])),
                    f64p(Int(ctx[unsafe_offset=C_RW])),
                    level, ww, shortcut, dw, stampw,
                    tag, rev_buf, ocount, whk, whv, whs,
                )
                var has_witness = False
                var k = 0
                while k < fwd_n:
                    var x = fwd_buf[unsafe_offset=k]
                    if stampw[unsafe_offset=x] == tag:
                        if du[unsafe_offset=x] + dw[unsafe_offset=x] < shortcut:
                            has_witness = True
                            break
                    k += 1
                if not has_witness:
                    var ok = fs_add(
                        fhead, fnxt, fto, fw,
                        i32p(Int(ctx[unsafe_offset=C_FMID])),
                        used, uu, ww, shortcut, v,
                    )
                    if not ok:
                        used[unsafe_offset=2] = used[unsafe_offset=0]
                        return -3
                    ok = fs_add(
                        i32p(Int(ctx[unsafe_offset=C_RHEAD])),
                        i32p(Int(ctx[unsafe_offset=C_RNXT])),
                        i32p(Int(ctx[unsafe_offset=C_RTO])),
                        f64p(Int(ctx[unsafe_offset=C_RW])),
                        i32p(Int(ctx[unsafe_offset=C_RMID])),
                        used, ww, uu, shortcut, v,
                    )
                    if not ok:
                        used[unsafe_offset=2] = used[unsafe_offset=0]
                        return -3
                    shortcuts += 1
                b += 1
            a += 1
    return shortcuts


@export("gh_ch_prepare")
def gh_ch_prepare(n: Int, ctx_addr: Int) abi("C") -> Int:
    """Build contraction hierarchies. Returns the number of shortcuts added.

    A negative return is an out-of-capacity code: -2 for the base-graph seeding
    pass, -3 while inserting shortcuts. In both cases slot 2 of the `used`
    block is set to the slot counter at the point of failure so the caller can
    see how far it got. The caller grows the buffers and retries, so a partial
    contraction is never handed back as a usable CH graph.
    """
    return _contract(n, i64p(ctx_addr))


@export("gh_ch_middle")
def gh_ch_middle(n: Int, src: Int, dst: Int, ctx_addr: Int) abi("C") -> Int:
    """Two complete upward searches; return the middle node, or -1.

    The shortest path in a CH graph rises to a peak and falls again, so the
    peak is upward-reachable from both ends; the cheapest node common to both
    upward closures therefore lies on a shortest path and `df[m] + dt[m]` is the
    true distance. Returns -1 when the two closures do not meet, i.e. when the
    target is unreachable.
    """
    var ctx = i64p(ctx_addr)
    var fhead = i32p(Int(ctx[unsafe_offset=C_FHEAD]))
    var fnxt = i32p(Int(ctx[unsafe_offset=C_FNXT]))
    var fto = i32p(Int(ctx[unsafe_offset=C_FTO]))
    var fw = f64p(Int(ctx[unsafe_offset=C_FW]))
    var rhead = i32p(Int(ctx[unsafe_offset=C_RHEAD]))
    var rnxt = i32p(Int(ctx[unsafe_offset=C_RNXT]))
    var rto = i32p(Int(ctx[unsafe_offset=C_RTO]))
    var rw = f64p(Int(ctx[unsafe_offset=C_RW]))
    var level = i32p(Int(ctx[unsafe_offset=C_LEVEL]))
    var df = f64p(Int(ctx[unsafe_offset=M_DF]))
    var dt = f64p(Int(ctx[unsafe_offset=M_DT]))
    var sf = i32p(Int(ctx[unsafe_offset=M_SF]))
    var st = i32p(Int(ctx[unsafe_offset=M_ST]))
    var hk = f64p(Int(ctx[unsafe_offset=M_HK]))
    var hv = i32p(Int(ctx[unsafe_offset=M_HV]))
    var hs = i64p(Int(ctx[unsafe_offset=M_HS]))

    for i in range(n):
        df[unsafe_offset=i] = BIG
        dt[unsafe_offset=i] = BIG
        sf[unsafe_offset=i] = Int32(0)
        st[unsafe_offset=i] = Int32(0)
    if src < 0 or src >= n or dst < 0 or dst >= n:
        return -1

    hs[unsafe_offset=0] = Int64(0)
    df[unsafe_offset=src] = Float64(0.0)
    sf[unsafe_offset=src] = Int32(1)
    heap_push(hk, hv, hs, Float64(0.0), Int32(src))
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = item[1]
        var du = item[0]
        if sf[unsafe_offset=u] == Int32(2):
            continue
        if du > df[unsafe_offset=u]:
            continue
        sf[unsafe_offset=u] = Int32(2)
        var lu = level[unsafe_offset=u]
        var e = Int(fhead[unsafe_offset=u])
        while e >= 0:
            var v = fto[unsafe_offset=e]
            if level[unsafe_offset=v] > lu:
                var nd = du + fw[unsafe_offset=e]
                if nd < df[unsafe_offset=v]:
                    df[unsafe_offset=v] = nd
                    sf[unsafe_offset=v] = Int32(1)
                    heap_push(hk, hv, hs, nd, v)
            e = Int(fnxt[unsafe_offset=e])

    hs[unsafe_offset=0] = Int64(0)
    dt[unsafe_offset=dst] = Float64(0.0)
    st[unsafe_offset=dst] = Int32(1)
    heap_push(hk, hv, hs, Float64(0.0), Int32(dst))
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = item[1]
        var du = item[0]
        if st[unsafe_offset=u] == Int32(2):
            continue
        if du > dt[unsafe_offset=u]:
            continue
        st[unsafe_offset=u] = Int32(2)
        var lu = level[unsafe_offset=u]
        var e = Int(rhead[unsafe_offset=u])
        while e >= 0:
            var v = rto[unsafe_offset=e]
            if level[unsafe_offset=v] > lu:
                var nd = du + rw[unsafe_offset=e]
                if nd < dt[unsafe_offset=v]:
                    dt[unsafe_offset=v] = nd
                    st[unsafe_offset=v] = Int32(1)
                    heap_push(hk, hv, hs, nd, v)
            e = Int(rnxt[unsafe_offset=e])

    var best = -1
    var best_cost = BIG
    var i = 0
    while i < n:
        if sf[unsafe_offset=i] == Int32(2):
            if st[unsafe_offset=i] == Int32(2):
                var c = df[unsafe_offset=i] + dt[unsafe_offset=i]
                if c < best_cost:
                    best_cost = c
                    best = i
        i += 1
    return best


@export("gh_ch_downward")
def gh_ch_downward(
    n: Int, middle: Int, reverse: Int, ctx_addr: Int
) abi("C") -> Int:
    """Dijkstra from `middle` over the CH graph, following only lower levels.

    This has to run on the contraction-hierarchy star, not on the base graph.
    The upward searches legitimately step onto shortcut edges, and a shortcut is
    not an edge of the base graph, so a downward walk over the base graph cannot
    retrace what the upward walk found. Here the search is on the star and
    `S_PREV` receives the star edge index that reached each node, which is what
    lets the Python layer unpack the shortcuts.

    The source-to-middle part of a shortest path rises in level, so walking it
    backwards from the middle strictly lowers the level at every step, and the
    same holds from the middle to the target, so each pass reaches its end at
    exactly the distance the upward search recorded. Returns 0 on a bad middle
    node.
    """
    var ctx = i64p(ctx_addr)
    var level = i32p(Int(ctx[unsafe_offset=C_LEVEL]))
    # reverse == 0 walks the forward star towards the source, reverse != 0 walks
    # the mirrored reverse star towards the target. Both halves of the route
    # have to be searched: on a directed graph the middle node often has no
    # outgoing forward edge at all, so a forward-only pass cannot reach the
    # target side.
    var fhead = i32p(Int(ctx[unsafe_offset=C_FHEAD]))
    var fnxt = i32p(Int(ctx[unsafe_offset=C_FNXT]))
    var fto = i32p(Int(ctx[unsafe_offset=C_FTO]))
    var fw = f64p(Int(ctx[unsafe_offset=C_FW]))
    if reverse != 0:
        fhead = i32p(Int(ctx[unsafe_offset=C_RHEAD]))
        fnxt = i32p(Int(ctx[unsafe_offset=C_RNXT]))
        fto = i32p(Int(ctx[unsafe_offset=C_RTO]))
        fw = f64p(Int(ctx[unsafe_offset=C_RW]))
    var dist = f64p(Int(ctx[unsafe_offset=S_DIST]))
    var prev_edge = i32p(Int(ctx[unsafe_offset=S_PREV]))
    var prev_node = i32p(Int(ctx[unsafe_offset=S_PREVN]))
    var done = u8p(Int(ctx[unsafe_offset=S_DONE]))
    var hk = f64p(Int(ctx[unsafe_offset=S_HK]))
    var hv = i32p(Int(ctx[unsafe_offset=S_HV]))
    var hs = i64p(Int(ctx[unsafe_offset=S_HS]))

    reset_dist(n, dist, prev_edge, done)
    for i in range(n):
        prev_node[unsafe_offset=i] = Int32(-1)
    hs[unsafe_offset=0] = Int64(0)
    if middle < 0 or middle >= n:
        return 0
    dist[unsafe_offset=middle] = Float64(0.0)
    heap_push(hk, hv, hs, Float64(0.0), Int32(middle))
    while Int(hs[unsafe_offset=0]) > 0:
        var item = heap_pop(hk, hv, hs)
        var u = Int(item[1])
        if done[unsafe_offset=u] == UInt8(1):
            continue
        done[unsafe_offset=u] = UInt8(1)
        var du = dist[unsafe_offset=u]
        var lu = level[unsafe_offset=u]
        var e = Int(fhead[unsafe_offset=u])
        while e >= 0:
            var v = Int(fto[unsafe_offset=e])
            if level[unsafe_offset=v] < lu:
                if done[unsafe_offset=v] == UInt8(0):
                    var nd = du + fw[unsafe_offset=e]
                    if nd < dist[unsafe_offset=v]:
                        dist[unsafe_offset=v] = nd
                        prev_edge[unsafe_offset=v] = Int32(e)
                        prev_node[unsafe_offset=v] = Int32(u)
                        heap_push(hk, hv, hs, nd, Int32(v))
            e = Int(fnxt[unsafe_offset=e])
    return 1
