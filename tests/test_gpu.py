"""The GPU phase decision is a CPU-only path, and stays honest.

The README's "GPU: CPU only" section rests on two measured facts: the geometry
kernels are below 2 flops/byte, and this toolchain cannot compile their
`Float64` trigonometry into a device kernel -- `std.math`'s `sin`, `cos` and
`tan` have no `Float64` device implementation, `asin`/`atan2` resolve through
libcall symbols that ptxas leaves unresolved, and the libdevice entry points
(`__nv_*`) are not linked either.

These tests never touch a device. They check that the shipped library exposes no
device entry point, and that the compiler still refuses the exact constructs the
README says it refuses -- so the first toolchain that lifts that limit turns
this red and forces the note to be revisited, instead of leaving a stale claim
in the README.

The refusal is a compile-time error. `mojo build` reports it without launching
anything, which is why `build` is used rather than `run`: instantiation for the
device happens during compilation, so the probe is never run, and the test does
not depend on how busy the shared GPU is. The kernel is `@export`ed so that
building the file instantiates it.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

from mojo_graphhopperpy import _lib

_HEADER = """
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
{imports}

comptime B: Int = 256


def kernel(a: Pointer[Float64, AnyOrigin[mut=True]],
           o: Pointer[Float64, AnyOrigin[mut=True]], n: Int32):
    var i = block_idx.x * B + thread_idx.x
    if i < Int(n):
        o.unsafe_store(Int(i), {expr})
{declare}

@export("trigger_instantiation")
def trigger_instantiation() abi("C"):
    try:
        var ctx = DeviceContext()
        var d = ctx.enqueue_create_buffer[DType.float64](4)
        var o = ctx.enqueue_create_buffer[DType.float64](4)
        ctx.enqueue_function[kernel](d, o, Int32(4), grid_dim=1, block_dim=B)
    except:
        pass
"""

# fn -> probe, and the diagnostic the README records for it.
_REFUSED = {
    "sin": dict(
        imports="from std.math import sin",
        expr="sin(a.unsafe_load(Int(i)))",
        declare="",
        expected="DType.float64 is not supported for sin on NVIDIA GPU",
    ),
    "cos": dict(
        imports="from std.math import cos",
        expr="cos(a.unsafe_load(Int(i)))",
        declare="",
        expected="DType.float64 is not supported for cos on NVIDIA GPU",
    ),
    "tan": dict(
        imports="from std.math import tan",
        expr="tan(a.unsafe_load(Int(i)))",
        declare="",
        expected="failed to run the pass manager for offload functions",
    ),
    "asin": dict(
        imports="from std.math import asin",
        expr="asin(a.unsafe_load(Int(i)))",
        declare="",
        expected="no libcall available for fasin",
    ),
    "atan2": dict(
        imports="from std.math import atan2",
        expr="atan2(a.unsafe_load(Int(i)), 1.0)",
        declare="",
        expected="Unresolved extern function 'atan2'",
    ),
    "libdevice_nv_sin": dict(
        imports="",
        expr="nv_sin(a.unsafe_load(Int(i)))",
        declare='@extern("__nv_sin")\ndef nv_sin(x: Float64) abi("C") -> Float64:\n    ...',
        expected="Unresolved extern function '__nv_sin'",
    ),
}


def _mojo() -> str:
    exe = shutil.which("mojo")
    if exe is None:
        candidate = pathlib.Path(
            os.environ.get("CONDA_PREFIX", "")
        ) / "bin" / "mojo"
        exe = str(candidate) if candidate.exists() else None
    if exe is None:
        pytest.skip("no `mojo` on PATH; run the suite with `pixi run test`")
    return exe


def test_library_exports_no_gpu_entry_point():
    """The shipped shared library is CPU-only: no symbol promises a device path."""
    assert not [name for name in _lib._SIGNATURES if "gpu" in name.lower()]


@pytest.mark.parametrize("fn", sorted(_REFUSED))
def test_float64_device_trig_still_unsupported(tmp_path, fn):
    """The README names these refusals; if one starts compiling, revisit it."""
    probe = _REFUSED[fn]
    src = tmp_path / f"probe_{fn}.mojo"
    src.write_text(_HEADER.format(**probe))
    proc = subprocess.run(
        [_mojo(), "build", "--emit", "shared-lib", str(src),
         "-o", str(tmp_path / f"probe_{fn}.so")],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0:
        pytest.fail(
            f"the toolchain now compiles Float64 `{fn}` into a device kernel. "
            "That lifts the blocker recorded in the README's GPU section: "
            "re-count the intensity numbers, measure a GPU path against the "
            "CPU path at the sizes callers use, and update the README."
        )
    assert probe["expected"] in combined, (
        f"`{fn}` failed for a different reason than the README records; "
        f"the GPU note is stale:\n{combined[-2000:]}"
    )
