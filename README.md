# mojo-graphhopper-py

Mojo implementations of the routing algorithms at the heart of
[GraphHopper](https://github.com/graphhopper/graphhopper): great-circle
geometry, Dijkstra, A*, and contraction hierarchies with witness search,
shortcut expansion and a two-sided upward/downward query.

```python
import mojo_graphhopperpy as mgh

g = mgh.grid_graph(40, 40)                 # a lat/lon grid with a slow column
ch = g.prepare_ch()                        # contraction hierarchies
p = ch.route(0, 1200)                      # -> Path(nodes, distance, time, points)
p.distance, p.time                         # metres, seconds
```

The Python package is `mojo_graphhopperpy`; it imports alongside `routingpy`
and `graphh` and shadows no module name of theirs.

## What upstream package is this a port of?

**There is no Python package called `graphhopper` or `graphhopper-py` to port.**
Verified: `https://pypi.org/pypi/graphhopper`, `/graphhopper-py`,
`/graphhopperpy`, `/graphhopper_python` and `/pygraphhopper` all return HTTP
404, and `graphhopper` is not in the shared test venv. GraphHopper itself is a
**Java** routing engine; the only Python distributions that talk to it are HTTP
clients. `routingpy.Graphhopper` (which *is* in the test venv, and which this
package deliberately does not overlap with — that is `mojo-routingpy`'s
subject) is 640 lines of `requests` and JSON parsing with no arithmetic at all.

So this port does the honest thing: it ports the **algorithms that engine is
built on**, which is where the numerics actually are, and states plainly that
the Python surface being replaced is an HTTP client. The algorithms are the
ones named in GraphHopper's own `docs/core/routing.md`: `dijkstra`, `astar`,
bidirectional search, and contraction hierarchies for the speed mode.

## Covered subset

| area | implemented API |
| --- | --- |
| Geometry | `haversine`, `equirectangular`, `bearing`, `edge_weights` — GraphHopper's `DistanceEarthProjection` and `DistancePlaneProjection`; parallel above 262144 items |
| Weighting | `Graph` with per-edge speeds; weight is length / speed, as in `DistanceWeighting` |
| Flexible mode | `Graph.dijkstra`, `Graph.astar` with a great-circle heuristic |
| Speed mode | `Graph.prepare_ch` → `CHGraph` (greedy contraction, bounded witness search, shortcut insertion) and `CHGraph.route` (two upward searches, middle node, two downward searches, recursive shortcut expansion) |
| Test aids | `grid_graph`, `reference_shortest_path` (an independent Bellman-Ford oracle) |

**Not implemented, and not faked:** OSM import, the HTTP Directions API, JSON
request/response handling, landmark (LM/ALT) preprocessing and its heuristic,
custom weight models and turning instructions, `isochrones`, `matrix`,
`details`, and subnetwork extraction. Those are IO, configuration and
preprocessing strategies rather than numeric kernels; the real GraphHopper
server and the Python clients are where they belong. `CHGraph.route` returns a
node path with distance and time, not a turn-by-turn instruction list.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-graphhopper-py.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task.

## Performance

Best-of-N wall clock in one process, every case gated on numerical agreement
first. The routing baselines are `networkx`, the fastest reasonable pure-Python
implementation available here, on the same graph and the same query list. That is
the same *input*, not the same *work*: the networkx calls below run whole-graph
`single_source_dijkstra_path_length`, which does not stop at the target, while
the ported `dijkstra` does -- and the CH rows compare networkx Dijkstra against a
contraction hierarchy, two different algorithms. The A* row is the only
like-for-like one, because both sides are this package. Geometry is against the
fastest vectorised NumPy formulation.

| case | reference | mojo-graphhopper-py | result |
| --- | ---: | ---: | ---: |
| haversine n=1048576 | 134.58 ms | 38.70 ms | 3.48x faster |
| dijkstra 60x60, 20 queries | 61.32 ms | 6.85 ms | 8.95x faster |
| astar 60x60, 20 queries | 6.89 ms (Dijkstra) | 12.76 ms | **0.54x, slower** |
| ch 40x40, 200 queries | 134.12 ms | 240.47 ms | **0.56x, slower** |
| ch 40x40, 200 queries + preprocessing | 134.12 ms | 1706.81 ms | **0.08x, much slower** |

These numbers were re-measured in a verification pass on a shared 72-core host
under load (load average 24-42 over the pass); three consecutive `pixi run
bench` runs moved haversine's reference by 12% and the Dijkstra ratio between
7.9x and 9.5x. One later run, at a load average of 36, put the ported haversine
at 90.85 ms, more than twice every other measurement of it (38.70-42.70 ms in
five runs); it is reported as host noise rather than discarded, because the
median of the runs above is what the table carries. `/tmp/mojo-bench.lock` is
shared with other repositories, so a run can also queue for tens of minutes
behind another project's benchmark -- that is wall-clock waiting, not timing.
Within a run every pair is the same graph and the same query list, but the
absolute milliseconds are this host on that day, not a portable score.
`bench/verify.md` records the earlier figures these replace and why.

Reported as losses, because they are:

- **A* is slower than Dijkstra here.** On a grid a great-circle bound is weak —
  the settled counts are 33422 against Dijkstra's 33450 out of 7200 nodes per
  query — so the heuristic buys nothing and costs a full haversine evaluation
  per query on the Python side. A* wins on geometric graphs where the bound is
  tight; it loses on grids, and this table measures the case where it loses.
- **Contraction hierarchies lose on 200 queries.** Preprocessing a 40x40 grid
  costs about 1.7 s of witness searches (measured this pass: 1655-1927 ms across
  runs), and 200 queries are nowhere near enough to amortise it. CH exists to
  serve millions of queries on a continent-sized graph; the crossover is far
  outside anything this benchmark measures, so the honest number here is the
  loss.
- Dijkstra's 8.9x is the number that reflects what this port is for: the same
  pointer-chasing loop that networkx runs in Python, with the heap and the
  relaxations compiled. The margin is larger than the milliseconds suggest,
  because the networkx side here runs `single_source_dijkstra_path_length`,
  which settles the whole graph instead of stopping at the target: it does more
  work per query than the ported `dijkstra`, so this row is not an equal-work
  comparison.

Reproduce with:

```bash
pixi run bench
```

The haversine row moved from 155.82 ms to the low 40s of milliseconds when the
elementwise geometry kernels were parallelised; before/after and thread-scaling
numbers are in "Parallelism" below, and so are the paths that were measured and
*not* parallelised. (The 155.82 ms "before" figure is from the parallel-phase
work and was not re-measured in the verification pass; the after-side 39.17 ms
it was paired with did reproduce within 20%, and the interleaved A/B below was
re-run from the pre-parallel source.)

## Parallelism

The geometry kernels (`haversine`, `equirectangular`, `bearing`,
`edge_weights`) split their work over threads. Every other kernel is serial,
and that is a measured decision, not an oversight -- see "Not parallelised".
Nothing runs on a GPU; the gate that settles that, and the toolchain limit that
would have blocked it anyway, are in "GPU: CPU only" below.

`parallelize` comes from `max.algorithm` (the `std` module lost it; see
`MOJO_NOTES.md` section 4). That means the env pins `max` beside `mojo`:

```toml
mojo = "==1.2.0.dev2026092905"
max  = "==26.7.0.dev2026092905"
```

**Size threshold.** One work item is one coordinate pair. Below the threshold
the kernel runs the serial loop unchanged, so small callers are untouched.

| n | serial | 4 threads | speedup |
| ---: | ---: | ---: | ---: |
| 4096 | 0.42 ms | 0.42 ms | 1.00x |
| 16384 | 1.64 ms | 1.64 ms | 1.00x |
| 32768 | 3.26 ms | 3.26 ms | 1.00x |
| 65536 | 6.45 ms | 6.49 ms | 0.99x |
| 131072 | 13.00 ms | 13.06 ms | 1.00x |
| 262144 (2^18) | 26.06 ms | 12.61 ms | 2.07x |
| 524288 | 52.08 ms | 19.27 ms | 2.70x |
| 1048576 | 104.55 ms | 33.73 ms | 3.10x |

Best of 7, six interleaved rounds per size, measured in the verification pass.
Nothing below 2^18 beats serial: 65536 and 131072 measure at 0.99-1.00x, so at
that size dispatch and memory traffic consume the entire gain. The first size
where the win is clear and repeatable is **262144 (2^18)**: parallel came in
faster than serial in 6 of 6 rounds there, against 1-5 of 6 below it, which is
coin-flip noise. The shipped threshold is that measured crossover itself, not an
offset above it, because the sizes above it are where the gain is stable
(2.07x at 2^18, 2.70x at 2^19, 3.10x at 2^20); callers below it keep the serial
loop and pay nothing. (An earlier revision of this section called the crossover
"2.14x" and applied that figure to `524288` as well; 2.14x is the 4-thread
*equirectangular* number from the next table, and it was never this crossover's
speedup. Re-measured, the crossover is 2.07x and `524288` is 2.70x, below.)

**Threads.** Cap is 4 (`PAR_WORKER_CAP`), applied on top of whatever
`GH_WORKERS` asks for. The host is shared, so the cap is a contract rather
than a reading of `nproc`. Cases at n=1M, best-of-7, `bench/parallel.py`:

| case | 1 thread | 2 threads | 4 threads | 4x |
| --- | ---: | ---: | ---: | ---: |
| haversine | 106.78 ms | 60.68 ms | 34.40 ms | 3.10x |
| equirectangular | 20.28 ms | 17.43 ms | 11.99 ms | 1.69x |
| bearing | 121.58 ms | 68.73 ms | 37.32 ms | 3.26x |

`haversine` and `bearing` are close to linear between 2 and 4 threads (1.76x and
1.77x at 2 threads, 3.10x and 3.26x at 4). `equirectangular` is the weakest: it
barely moves between 1 and 2 threads (1.16x) and then jumps to 1.69x at 4, so it
is the one case that is *not* linear across the range. Its serial cost is a
quarter of `haversine`'s, so dispatch is a larger share of each chunk; it is
still a clear win at the shipped threshold. The cap is kept at 4 anyway -- going
past it needs a benchmark on an idle host, and this host is not idle.

Run-to-run spread on this shared host is real and larger than the differences
between the first two columns for `equirectangular`: a second pass measured the
same three kernels at 111.96 / 67.42 / 43.80 ms (2.56x), 25.93 / 18.05 /
14.06 ms (1.84x) and 131.50 / 75.82 / 45.14 ms (2.91x). Read the 4-thread
column as 2.9-3.1x for `haversine`, 1.7-1.8x for `equirectangular` and 2.9-3.3x
for `bearing`, not as three-significant-figure results.

**Before and after, one build against the previous one.** The same script times
the pre-parallel build (`git show HEAD:src/kernels.mojo`, built separately) and
the current build alternately, round by round, so shared-host noise lands on
both columns:

| case | before | after | before/after |
| --- | ---: | ---: | ---: |
| haversine n=1M | 106.11 ms | 35.38 ms | 3.00x |
| equirectangular n=1M | 19.80 ms | 12.64 ms | 1.57x |
| bearing n=1M | 126.09 ms | 45.63 ms | 2.76x |
| edge_weights n=1M | 113.03 ms | 37.98 ms | 2.98x |
| haversine n=2^18 | 26.36 ms | 14.88 ms | 1.77x |
| haversine n=65536 | 6.67 ms | 6.86 ms | 0.97x |

`bench/ab.py OLD_DIR . --reps 9`, re-run in the verification pass from the
pre-parallel kernel source. A second identical pass landed within 13% on every
row (haversine 2.86x, equirectangular 1.60x, bearing 2.85x, edge_weights 2.77x,
2^18 2.04x, 65536 1.05x), and the direction is the same in all six rows in both
passes.

The last row is the threshold doing its job: 65536 is below 2^18, so both
columns run the identical serial loop, and the few-percent difference there
(3% down in one pass, 5% up in the other) is host noise on a shared machine,
not an effect of this change.

**Identical results, not merely close.** Each kernel's scalar body is a single
`@always_inline def` called from both the serial loop and the chunk worker, and
chunks are contiguous index ranges that meet exactly at n. `tests/test_parallel.py`
runs both paths on the same input at 1000, 2^18, 2^18 + 7 and 2^20 items and
requires bit-identical output; it also checks 2 and 3 workers against serial,
the threshold boundary either side, and that `GH_WORKERS=64` does not exceed the
cap.

**Not parallelised, with the measurement.** These were timed at the worker
counts that would be used, and none of them gained:

| case | 1 thread | 2 threads | 4 threads | decision |
| --- | ---: | ---: | ---: | --- |
| `dijkstra` 60x60, 200 queries | 73.73 ms | 73.49 ms | 73.43 ms | serial |
| `CHGraph.route` 60x60, 200 queries | 708.62 ms | 684.95 ms | 717.45 ms | serial |
| `prepare_ch` 60x60 | 16095 ms | 15722 ms | 15856 ms | serial |

- **`dijkstra` and `route` batches** share one scratch block per graph
  (`_scratch`), so parallel queries need a scratch block per worker. Query
  halves of these timings are dominated by the Python half of a query (the
  predecessor walk and path finish), which does not parallelise in Mojo at all,
  and 200 queries of a 3600-node grid is 2.2 ms per 200 in the kernel. The
  measured ceiling is the 1.00x above. No parallel path was shipped for them.
- **`prepare_ch`**: the greedy contraction is order-dependent, so parallel
  contraction changes *which* shortcuts are added and therefore the hierarchy.
  It is parallelisable in principle -- the witness-search scratch is
  worker-local -- but the parallel prepare built for that was reverted: the
  scratch blocks for it are per-worker copies of four n-sized index buffers and
  two n-sized distance buffers, sized from a capacity loop in the Python layer,
  and the first version of that loop guessed a cap instead of growing until the
  kernel reported success. Shipping it without the growth loop would have
  silently returned a `-2` capacity error on larger graphs. Reverted rather
  than shipped broken; the serial prepare is unchanged.

Reproduce the numbers above:

```bash
pixi run python bench/parallel.py --sizes     # serial/parallel crossover
pixi run python bench/parallel.py             # 1 / 2 / 4 threads
pixi run python bench/ab.py OLD_DIR .         # before/after, interleaved
```

`GH_WORKERS=N` lowers the ceiling for one process; values above 4 are clamped
down to 4, and `GH_WORKERS=1` runs the serial loop.

## GPU: CPU only, for a measured reason

There is no GPU path in this port. That is a decision backed by two things: the
arithmetic intensity threshold, and a toolchain limit that was probed, not
assumed.

**Gate 1, intensity.** `bench/analysis.md` does not exist in this checkout, so
the intensity was counted from the scalar bodies in `src/kernels.mojo`
(`_hav_one`, `_equirect_one`, `_bearing_one`, `_edge_weight_one`). Each
elementwise call reads 4 arrays and writes 1, all `float64`, so 40 bytes per
pair; `edge_weights` reads a fifth and writes one, 48 bytes.

| kernel | flops/pair | bytes/pair | flops/byte |
| --- | ---: | ---: | ---: |
| `haversine` | 29 | 40 | 0.73 |
| `equirectangular` | 21 | 40 | 0.53 |
| `bearing` | 30 | 40 | 0.75 |
| `edge_weights` | 30 | 48 | 0.63 |

Counted from the bodies in `src/kernels.mojo` as multiply/add/divide and one
flop per libm call: `haversine` is 3 scales, 3 differences, an FMA, 2 `sin`,
2 `cos`, 4 products, 2 clamps, `sqrt`, `asin` and a final product, 29;
`equirectangular` is 2 scales, 2 differences, an add, a scale, 4 products and
`sqrt`, 21; `bearing` is 3 scales, a difference, 4 `sin`/`cos`, 2 products, a
subtraction, `atan2`, a scale and a wrap, 30; `edge_weights` is `haversine`'s
29 plus a divide, 30.

All four are **below 2 flops per byte**, so the gate stops them: the verdict is
CPU only because the kernels are memory-bound and a device round trip cannot pay
for itself at any size. They are also called at sizes that make it worse: a
60x60 grid calls `haversine` once on `m = 28320` directed edges, a 40x40 grid on
12480.

The routing kernels are further below it -- `dijkstra`, `astar` and the CH
queries are pointer-chasing graph traversals, one load per visited node, with no
reusable arithmetic.

**Gate 2, the shared device.** `nvidia-smi --query-gpu=memory.free` reported
`4559 MiB` free on an RTX 5090, so the memory gate passed; it is not the reason.

**The toolchain cannot compile these kernels for the device either.** Every line
below was produced by a minimal `max.gpu` kernel on the pinned toolchain
(`mojo 1.2.0.dev2026092905`), refusing at compile time:

| attempt | result |
| --- | --- |
| `sin` on `Float64` in a device kernel | `error: constraint failed: DType.float64 is not supported for sin on NVIDIA GPU` |
| `cos` on `Float64` | same, for `cos` |
| `asin` on `Float64` | `error: no libcall available for fasin` |
| `atan2` on `Float64` | `ptxas fatal: Unresolved extern function 'atan2'` |
| `tan` on `Float64` | same offload-pass failure as `sin` |
| `sin`/`cos` inside a plain host fallback branch of the same function | still fails: the offload pass runs on the whole function |
| `@extern("__nv_sin")`, `@extern("sin")` | `ptxas fatal: Unresolved extern function` (libdevice is not linked) |
| `sin` on `Float32` | compiles and runs |

The failures are consistent when the probes are spaced apart: `sin`, `cos`,
`tan`, `asin` and `__nv_sin` each reproduced on every one of 2-3 separate runs.
The one run-to-run difference seen was a bare `Float64` kernel with no trig in
it (which otherwise builds and runs) reporting `Unresolved extern` twice while
the device was saturated, and passing again once the load dropped. The refusal
of the trig calls themselves did not vary.

`haversine`, `equirectangular`, `bearing` and `edge_weights` are built out of
trigonometry on `Float64`. Dropping to `Float32` would change the results these
kernels return -- the geometry tests hold the `Float64` path to `rtol=1e-12`,
and `Float32` trigonometry cannot meet that -- so a `Float32` GPU path would be
a different, less accurate API, not an accelerated one.

So the elementwise kernels fall out on the intensity gate before the compiler is
reached, and the compiler would have refused them anyway. What did compile and
run on this host was verified end to end (`DeviceContext`, `enqueue_copy`,
`enqueue_function`, `enqueue_create_buffer`, `synchronize` all work, and tail
predication with `if i < n` was checked); no kernel reached the point of being
worth timing. There are no before/after GPU numbers to report because no GPU
path was built, and none are invented here.

The CPU kernels stay as they are, including the 4-thread parallel path above.
`tests/test_gpu.py` checks that a CPU host is the shipped behaviour: it asserts
no GPU entry point is exported from the shared library, and re-compiles a
`Float64` device probe for each construct in the table above, so the first
toolchain release that lifts the limit fails the suite -- with a message saying
to measure a GPU path -- rather than leaving this note stale. The probe is built
with `mojo build`, which reports the refusal at compile time without launching
anything, so the test needs no device and is not skipped for one.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-graphhopper-py.so`.

The Python layer owns every array — the CSR graph, the search scratch, the
forward-star adjacency and its capacity-retry loop. Mojo owns the loops: the
binary heap, the relaxations, the contraction order, the bounded witness
searches, and the upward and downward searches.

Elementwise geometry is split over up to 4 workers with `parallelize` from
`max.algorithm`, above a size threshold measured on this host. Routing kernels
stay serial: the benchmark at 1, 2 and 4 threads shows no gain for them, and
the reasons and numbers are in "Parallelism" above.

Buffers cross the C ABI as 64-bit addresses inside a caller-owned `Int64`
argument block, so no export has more than four parameters. That is not
stylistic: see the ABI note below.

### ABI note: exports keep their compared scalars in registers

On this toolchain (`mojo 1.2.0.dev2026092605`) an `@export ... abi("C")`
function with seven or more parameters miscompiles integer comparisons on the
arguments the SysV ABI passes on the stack. A probe over parameter counts 6..20
found the six register-passed arguments reliable and every count from 7 upwards
intermittently evaluating a two-term `or` of comparisons as true —
nondeterministically, and differently in two files with identical bodies.

So `gh_dijkstra` used to take `(n, head, to, w, src, dst, ...)` with `dst` in
the sixth slot and worked, while `gh_astar` needed one more argument, put `dst`
in the seventh, and then reported a bad node index on every single query. The
fix is structural: every scalar an export compares is declared in the first six
parameter slots, everything after them is an address that is only ever
dereferenced, and the addresses themselves live in an argument block whose
address is one of the four parameters.

The argument block's four groups occupy disjoint slot ranges. Packing them all
against zero once made `S_HEAD` and `C_FHEAD` the same slot; a CH prepare
wrote both and the second write won, so the kernel read the forward-star head
where it expected the CSR head and ran off the end of it.

The geometry exports still take exactly `(n, ctx)`: they read the worker count
from `GH_WORKERS` inside the entry point. That is not a fifth parameter smuggled
in through the environment -- it is what keeps the parallel path from changing
the ABI the routing kernels are built around.

### Numerical contract

Edge weights are `haversine / speed`, so every weight is a sum of lengths
divided by speeds. Mojo emits FMA, so the running sums differ from NumPy's in
the last bits and the parity tests assert tolerances, never bit equality. The
grid tests are checked at `rel=1e-9` against an independent Bellman-Ford; the
geometry tests are exact to `rtol=1e-12`, which is the honest bound for a
sphere computation where `std.math`'s `asin`/`sin`/`cos` are not glibc's.

The serial and parallel geometry paths are held to a stricter rule: exactly the
same output, bit for bit. They share one scalar body (`_hav_one`,
`_equirect_one`, `_bearing_one`, `_edge_weight_one`) and split only the index
range, so worker count can never move a value. `tests/test_parallel.py` enforces
`np.array_equal` between them, which is what makes a multi-threaded geometry
call safe to pass straight into the tolerance-based tests above.

`tests/test_routing.py` includes a check that runs a kernel on buffers mapped
above 4 GiB. A ctypes `argtypes` entry that is `c_int` instead of `c_int64` is
silent at every small-array size and only truncates there.

## License

MIT
