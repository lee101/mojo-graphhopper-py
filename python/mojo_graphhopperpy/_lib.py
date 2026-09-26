"""ctypes bridge to the compiled Mojo routing kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address inside a caller-owned Int64 "argument block", because an
`@export ... abi("C")` function with seven or more parameters miscompiles
integer comparisons on the stack-passed ones on this toolchain; see the ABI note
at the top of `src/kernels.mojo`. Keeping every export at four parameters is
what makes the argument lists below reliable.

The slot tables are the single source of truth for the layout, mirroring the
`comptime` slot numbers in the Mojo source.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-graphhopper-py.so"

# name -> (parameter kinds, return kind). "i" is Int, "o" is an argument-block
# address (Int), "f" is Float64, None means void.
_SIGNATURES = {
    "gh_haversine": (["i", "o"], None),
    "gh_equirect": (["i", "o"], None),
    "gh_bearing": (["i", "o"], None),
    "gh_edge_weights": (["i", "o"], None),
    "gh_dijkstra": (["i", "i", "i", "o"], "i"),
    "gh_astar": (["i", "i", "i", "o"], "i"),
    "gh_ch_prepare": (["i", "o"], "i"),
    "gh_ch_middle": (["i", "i", "i", "o"], "i"),
    "gh_ch_downward": (["i", "i", "i", "o"], "i"),
}

_CT = {"i": ctypes.c_int32, "f": ctypes.c_double, "o": ctypes.c_int64}

# Argument-block slot numbers, mirroring the Mojo `comptime` constants. The
# four groups occupy disjoint ranges; packing them against zero once made
# S_HEAD and C_FHEAD the same slot, and a CH prepare filled the block with both.
G_BASE, S_BASE, C_BASE, M_BASE = 0, 8, 32, 64
CTX_SLOTS = 72

G_LAT_A, G_LON_A, G_LAT_B, G_LON_B, G_OUT, G_SPEED = range(G_BASE, G_BASE + 6)
(
    S_HEAD, S_TO, S_W, S_DIST, S_PREV, S_DONE, S_HK, S_HV, S_HS, S_HEUR,
) = range(S_BASE, S_BASE + 10)
S_PREVN = S_BASE + 10
(
    C_FHEAD, C_FNXT, C_FTO, C_FW, C_FMID,
    C_RHEAD, C_RNXT, C_RTO, C_RW, C_RMID,
    C_USED, C_LEVEL, C_MAXLVL, C_NBCNT,
    C_DU, C_DW, C_STAMPU, C_STAMPW,
    C_ORDER, C_OCOUNT,
    C_OHK, C_OHV, C_OHS,
    C_WHK, C_WHV, C_WHS,
) = range(C_BASE, C_BASE + 26)
M_DF, M_DT, M_SF, M_ST, M_HK, M_HV, M_HS = range(M_BASE, M_BASE + 7)


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, (kinds, ret) in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = [_CT[k] for k in kinds]
        assert len(fn.argtypes) == len(kinds), f"{name}: argtypes arity"
        fn.restype = _CT[ret] if ret else None
    return lib


lib = _load()


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


class Args:
    """A reusable Int64 argument block.

    One block per call site, filled and refilled in place, so a query does not
    allocate. `slots` indexes the layout; values are buffer addresses.
    """

    __slots__ = ("buf",)

    def __init__(self, slots: int = CTX_SLOTS):
        self.buf = np.zeros(slots, dtype=np.int64)

    def __int__(self) -> int:
        return int(self.buf.ctypes.data)

    def set(self, pairs) -> int:
        for slot, value in pairs:
            self.buf[slot] = value
        return int(self.buf.ctypes.data)
