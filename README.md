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
| Geometry | `haversine`, `equirectangular`, `bearing`, `edge_weights` — GraphHopper's `DistanceEarthProjection` and `DistancePlaneProjection` |
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
implementation of the same algorithms available here, on the same graph and the
same query list. Geometry is against the fastest vectorised NumPy formulation.

| case | reference | mojo-graphhopper-py | result |
| --- | ---: | ---: | ---: |
| haversine n=1048576 | 281.81 ms | 155.82 ms | 1.81x faster |
| dijkstra 60x60, 20 queries | 239.15 ms | 16.51 ms | 14.48x faster |
| astar 60x60, 20 queries | 19.06 ms (Dijkstra) | 27.09 ms | **0.70x, slower** |
| ch 40x40, 200 queries | 440.27 ms | 509.37 ms | **0.86x, slower** |
| ch 40x40, 200 queries + preprocessing | 440.27 ms | 5848.03 ms | **0.08x, much slower** |

Reported as losses, because they are:

- **A* is slower than Dijkstra here.** On a grid a great-circle bound is weak —
  the settled counts are 33422 against Dijkstra's 33450 out of 7200 nodes per
  query — so the heuristic buys nothing and costs a full haversine evaluation
  per query on the Python side. A* wins on geometric graphs where the bound is
  tight; it loses on grids, and this table measures the case where it loses.
- **Contraction hierarchies lose on 200 queries.** Preprocessing a 40x40 grid
  costs about 2.8 s of witness searches, and 200 queries are nowhere near
  enough to amortise it. CH exists to serve millions of queries on a
  continent-sized graph; the crossover is far outside anything this benchmark
  measures, so the honest number here is the loss.
- Dijkstra's 14x is the number that reflects what this port is for: the same
  pointer-chasing loop that networkx runs in Python, with the heap and the
  relaxations compiled.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-graphhopper-py.so`.

The Python layer owns every array — the CSR graph, the search scratch, the
forward-star adjacency and its capacity-retry loop. Mojo owns the loops: the
binary heap, the relaxations, the contraction order, the bounded witness
searches, and the upward and downward searches.

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

### Numerical contract

Edge weights are `haversine / speed`, so every weight is a sum of lengths
divided by speeds. Mojo emits FMA, so the running sums differ from NumPy's in
the last bits and the parity tests assert tolerances, never bit equality. The
grid tests are checked at `rel=1e-9` against an independent Bellman-Ford; the
geometry tests are exact to `rtol=1e-12`, which is the honest bound for a
sphere computation where `std.math`'s `asin`/`sin`/`cos` are not glibc's.

`tests/test_routing.py` includes a check that runs a kernel on buffers mapped
above 4 GiB. A ctypes `argtypes` entry that is `c_int` instead of `c_int64` is
silent at every small-array size and only truncates there.

## License

MIT
