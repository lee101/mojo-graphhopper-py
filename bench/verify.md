# Verification pass

What was re-measured, what was not, and what changed. Host: 72 logical cores,
shared, load average 22-42 during the pass (`uptime` recorded around each run).
Every number below was produced in this session by a command named here.

## Gate

| command | result |
| --- | --- |
| `pixi run build && pixi run test` | green, 60 passed in 36.90 s |
| `pixi run bench` | **failed**: `ModuleNotFoundError: No module named 'networkx'` |

`pixi run bench` could not have produced the README table as checked in:
`networkx` was in neither `pixi.toml` nor `pixi.lock`, so it died after the first
row. Added with `pixi add networkx` (>=3.7,<4; resolved 3.7) and the lock
updated. Without that, the routing rows are unreproducible by anyone.

Final state after the README edits: `pixi run build && pixi run test` green
(61 passed, 1 xfailed -- see "Reproduced bug" below).

## Performance table (`pixi run bench`)

Three consecutive runs plus an independent harness (`bench/verify_bench.py`,
since deleted) that reimplements the same five cases. All agreed within host
noise. The README now carries the third `pixi run bench` run:

| case | README had | measured this session | verdict |
| --- | --- | --- | --- |
| haversine n=1048576 | 281.81 / 39.17 ms | ref 132.90-148.13, mojo 38.70-42.27 | reference 2x too slow; ratio 7.19x -> 3.48x |
| dijkstra 60x60 x20 | 239.15 / 16.51 ms | ref 54.69-89.84, mojo 6.85-9.45 | both columns; ratio 14.48x -> 8.95x |
| astar 60x60 x20 | 19.06 / 27.09 ms, 0.70x | dij 6.89, astar 12.76, 0.54x | both columns, same direction (slower) |
| ch 40x40 x200 | 440.27 / 509.37 ms, 0.86x | ref 134.12-149.73, ch 240.47-269.69, 0.50-0.57x | both columns, still slower |
| ch 40x40 x200 + prep | 440.27 / 5848.03 ms, 0.08x | ref 134.12-149.73, 1706.81-1927.36, 0.08x | both columns, still much slower |

The 20% rule would have forced corrections to the reference column of every row
and the mojo column of four of five. Direction of each row is unchanged: the
losses are still printed as losses, at numbers that are worse for the port in
relative terms than before (0.70x -> 0.54x, 0.86x -> 0.56x) -- not removed, not
reworded, and A* still says it is slower than Dijkstra.

One outlier, recorded rather than hidden: a later `pixi run bench` on the
newly built library reported the ported haversine at 90.85 ms (ratio 1.68x)
while the load average was 36. Every other measurement of that cell in this
session -- five bench runs and two harness runs -- was 38.52-42.70 ms. I could
not reproduce it and did not chase it further: `/tmp/mojo-bench.lock` is shared
with other repositories (`bench/try-lock.sh` exists for that reason), and a
re-measurement queued behind other projects' benchmarks for over 20 minutes
without starting. It is not in the table; the README notes it in prose so a
reader hitting the same outlier knows it was seen.

Prose that quoted replaced numbers was corrected: "about 2.8 s" of
preprocessing is now 1.7 s (measured 1655-1927 ms); "Dijkstra's 14x" is now 8.9x;
"2.25x" for the 2^18 crossover is now 2.07x; "2.14x" was a misapplied figure
(see below).

## Equal input, equal work

- Same input: every `bench.py` case builds the networkx graph from the same
  `Graph` (`_to_networkx`) and uses the same query list object on both sides.
  The harness did the same independently.
- Same work: **no**. The networkx side calls
  `single_source_dijkstra_path_length(G, s, t)`, which computes distances to
  every node and only *selects* `t` -- stopping at the target (as networkx's own
  `_dijkstra` supports with `target=`) would cut it short. The port's `dijkstra`
  does stop at `dst`. So the Dijkstra and CH rows measure the port against a
  baseline that settles the whole graph. The README now says this explicitly.
  It inflates the port's margin, so it does not rescue any of the losses.
- CH rows compare different algorithms (networkx Dijkstra vs CH with
  preprocessing). That was already the documented framing; left as is.
- No work is skipped on the Mojo side: each case first checks the answer against
  Bellman-Ford and fails before timing if it disagrees; `astar`'s settled counts
  (33422/33450) are printed.

## Parallelism section

- Size crossover (`bench/verify_sizes.py`, 6 interleaved rounds x best-of-7):
  serial 0.42/1.64/3.26/6.45/13.00/26.06/52.08/104.55 ms, 4 threads
  0.42/1.64/3.26/6.49/13.06/12.61/19.27/33.73 ms. Ported into the README.
  Two cells failed 20%: `262144` serial 31.57 -> 26.06, `524288` 4-thread
  24.43 -> 19.27. The crossover claim (nothing below 2^18 wins, 2^18 does)
  reproduced; the "earlier run measured 65536 at 1.20x" claim was **not**
  reproduced -- parallel never got below 0.99x at 65536 in 6 rounds -- and that
  sentence was replaced with what was measured.
- Threads (`bench/parallel.py`, best-of-7) plus a second pass: haversine
  3.10x/2.56x, equirectangular 1.69x/1.84x, bearing 3.26x/2.91x. The old
  equirectangular 2.14x did not reproduce (1.69x), and the sentence "the gain
  does not flatten at 4 ... all are still close to linear between 2 and 4" was
  false for equirectangular (1.16x at 2 threads, 1.69x at 4). Both corrected.
  Table and a stated run-to-run range replaced with measured values.
- Before/after (`bench/ab.py` against the pre-parallel kernel source built in
  `/tmp`, `--reps 9`, twice): every row within 13% of the README, so the table
  was replaced with the first pass and the second pass reported in text. The
  "9% difference" at 65536 was 3% and 5% (opposite signs) in the two passes;
  text corrected.
- Not-parallelised table (`bench/parallel.py`): 73.7/74.3/76.5 ms dijkstra,
  719/714/712 ms route, 16146/16032/16378 ms prepare at 1/2/4 threads, plus
  15885 vs 16559 ms prepare from three alternating rounds. All within 20% of the
  README. `prepare_ch` 16146 vs 15885 ms is the same measurement 1.6% apart, so
  the load moves `prepare_ch` by several percent; no cell changed.

## Tests: parity with upstream

`git log --stat -- tests/` shows three commits touching `tests/`, and all three
only add files or add lines: `b42789d` adds `conftest.py` and `test_routing.py`,
`37b0fb8` adds `test_parallel.py`, `f6d5d2b` adds `test_gpu.py`. No test file was
modified or deleted after it was added (the other three commits of the port,
including the "phase test" and "phase unroll" commits, touch no test file), so
no assertion was removed and no tolerance was loosened. `grep`-ing the history
for deleted `assert`/tolerance lines returns only added lines.

The parity assertions are intact and still compared against an independent
oracle: Bellman-Ford at `rel=1e-9`, geometry at `rtol=1e-12`. Nothing restored,
because nothing was lost.

The `phase test` commit did change code under a test-phase name: `fs_find` in
`src/kernels.mojo`. It returns the cheapest parallel edge instead of the first
one. Since that file is not under `tests/` it is not a hidden assertion change,
and it is the more correct behaviour -- kept.

## Reproduced bug, not fixed

While probing the CH rows I found a correctness bug that predates this pass and
that none of the existing tests catch:

```
0->1 (dear), 1->5->6->3->4 (cheap detour), 1->3 (dear direct edge)
ch.route(1, 4) = 472578.438239   Bellman-Ford = 83396.194983
```

The greedy contraction contracts node 1 while it still has an uncontracted
in-neighbour (5), so the two halves of the two-sided query disagree and the
query misses the detour. It reproduces identically on the pre-`fs_find` kernel
source (`git show 2310d0e:src/kernels.mojo`) and on HEAD, so this pass did not
introduce it. The grid tests and the random-graph test never produce that shape
(a contracted node with both a contracted and an uncontracted in-neighbour with
a cheaper detour), which is why the suite was green.

Added `test_ch_agrees_with_bellman_ford_when_a_detour_is_cheaper_than_an_edge`
in `tests/test_routing.py`, marked `xfail(strict=True)` with this cause. It
fails on the current build, the suite records it rather than hiding it, and a
future fix turns it into XPASS and breaks the build until the marker is removed.
Not fixed here: the rules for this pass allow code changes only to fix a bug
shown by a failing test, and the fix belongs in the contraction order, not in a
one-line patch.
