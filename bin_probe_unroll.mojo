"""Probe: what does `@parameter` do on mojo 1.2.0.dev2026092905?

Answers, by compiling:

- `@parameter for ...`: rejected (`'for' statement does not support decorators`).
- `@parameter var ...`: rejected (`'var' statement ... does not support decorators`).
- `@parameter def helper(k: Int)` called with a *loop-constant* index from an
  ordinary `while` loop: this is the shape that works, and it specialises the
  call per call site, which is what a hand-written unroll relies on.

Not part of the build.
"""

from std.math import asin, cos, sin, sqrt

comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime DEG2RAD: Float64 = 3.141592653589793 / 180.0
comptime EARTH_RADIUS_M: Float64 = 6371000.0
comptime UNROLL: Int = 4


@__parameter
def one_pair(
    lat_a: F64Ptr, lon_a: F64Ptr, lat_b: F64Ptr, lon_b: F64Ptr, res: F64Ptr, i: Int
):
    var la = lat_a.unsafe_load(i)
    var lb = lat_b.unsafe_load(i)
    var p1 = la * DEG2RAD
    var p2 = lb * DEG2RAD
    var dp = (lb - la) * DEG2RAD
    var dl = (lon_b.unsafe_load(i) - lon_a.unsafe_load(i)) * DEG2RAD
    var s1 = sin(dp * 0.5)
    var s2 = sin(dl * 0.5)
    var a = s1 * s1 + cos(p1) * cos(p2) * s2 * s2
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    res.unsafe_store(i, 2.0 * EARTH_RADIUS_M * asin(sqrt(a)))


def haversine_unrolled(
    n: Int, lat_a: F64Ptr, lon_a: F64Ptr, lat_b: F64Ptr, lon_b: F64Ptr, res: F64Ptr
):
    var i = 0
    while i + UNROLL <= n:
        one_pair(lat_a, lon_a, lat_b, lon_b, res, i + 0)
        one_pair(lat_a, lon_a, lat_b, lon_b, res, i + 1)
        one_pair(lat_a, lon_a, lat_b, lon_b, res, i + 2)
        one_pair(lat_a, lon_a, lat_b, lon_b, res, i + 3)
        i += UNROLL
    while i < n:
        one_pair(lat_a, lon_a, lat_b, lon_b, res, i)
        i += 1


def main() raises:
    var n = 5
    var la = List[Float64]()
    var lo = List[Float64]()
    var lb = List[Float64]()
    var lob = List[Float64]()
    for i in range(n):
        la.append(Float64(i))
        lo.append(Float64(i))
        lb.append(Float64(i) + 1.0)
        lob.append(Float64(i) + 1.0)
    var out = List[Float64](unsafe_uninit_length=n)
    haversine_unrolled(
        n,
        F64Ptr(unsafe_from_address=Int(la.unsafe_ptr())),
        F64Ptr(unsafe_from_address=Int(lo.unsafe_ptr())),
        F64Ptr(unsafe_from_address=Int(lb.unsafe_ptr())),
        F64Ptr(unsafe_from_address=Int(lob.unsafe_ptr())),
        F64Ptr(unsafe_from_address=Int(out.unsafe_ptr())),
    )
    print(out[0], out[4])
