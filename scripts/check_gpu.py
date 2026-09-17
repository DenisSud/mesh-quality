#!/usr/bin/env python3
"""GPU smoke test for this devenv: PyTorch, JAX and cuTile on the actual device.

Run it inside the dev shell:

    devenv shell -- check-gpu          # core checks
    devenv shell -- check-gpu --full   # + torch.compile / triton check

Exit status is 0 only if every check passes. Each check does real work on the
GPU (kernels, matmuls) and compares the result against the CPU, so a passing run
means the CUDA stack works, not just that the libraries import.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Callable

# Depth of the recursive `find` for bundled ELF executables (site-packages/*/bin,
# site-packages/*/*/bin, ...).
BIN_DIR_DEPTH = 4

# cuTile reads kernel annotations (ct.Constant[int]) from the function's *module*
# globals, so the import has to live at module level. Import failures are not
# fatal here: they are reported as a failed check.
try:
    import cuda.tile as ct
except Exception as exc:  # noqa: BLE001 - import errors are the diagnosis
    ct = None
    CUTILE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class Checker:
    """Collects check results and prints them as they run."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def section(self, title: str) -> None:
        print(f"\n\033[1m{title}\033[0m")

    def info(self, message: str) -> None:
        print(f"  {message}")

    def check(self, label: str, fn: Callable[[], str | None]) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - a failed check is the point
            self.failures.append(label)
            print(f"  \033[31mFAIL\033[0m {label}: {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            suffix = f"  {detail}" if detail else ""
            print(f"  \033[32m ok \033[0m {label}{suffix}")


def torch_tflops(size: int, seconds: float) -> float:
    return 2 * size**3 / seconds / 1e12


# --------------------------------------------------------------------------- #
# driver / environment
# --------------------------------------------------------------------------- #


def nvidia_smi(fields: list[str]) -> tuple[str, ...]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return tuple(part.strip() for part in out.splitlines()[0].split(","))


def report_driver(c: Checker) -> None:
    c.section("driver")
    name, driver, cap = nvidia_smi(["name", "driver_version", "compute_cap"])
    c.info(f"gpu       {name} (sm_{cap.replace('.', '')})")
    c.info(f"driver    {driver}")
    c.info(f"python    {sys.version.split()[0]}  ({sys.executable})")


# --------------------------------------------------------------------------- #
# bundled ELF executables (pip CUDA wheels ship binaries built for generic Linux)
# --------------------------------------------------------------------------- #


def elf_interpreter(path: Path) -> str | None:
    """Return the ELF PT_INTERP of `path`, or None if it is not a dynamic ELF."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"\x7fELF":
            return None
        fh.seek(0x10)
        _type, _machine, _version, _entry, phoff, _shoff, _flags, _ehsize, _phentsize, phnum = (
            struct.unpack("<HHIQQQIHHH", fh.read(42))
        )
        fh.seek(phoff)
        headers = [struct.unpack("<IIQQQQQQ", fh.read(56)) for _ in range(phnum)]
        for p_type, _flags, p_offset, _vaddr, _paddr, p_filesz, _memsz, _align in headers:
            if p_type != 3:  # PT_INTERP
                continue
            fh.seek(p_offset)
            return fh.read(p_filesz).split(b"\0", 1)[0].decode()
    return None


def bundled_elf_binaries() -> list[tuple[Path, str]]:
    roots = {sysconfig.get_paths()[key] for key in ("purelib", "platlib")}
    found: list[tuple[Path, str]] = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for depth in range(1, BIN_DIR_DEPTH + 1):
            for bin_dir in base.glob("/".join(["*"] * depth + ["bin"])):
                for exe in bin_dir.iterdir():
                    if not exe.is_file() or not exe.stat().st_mode & 0o111:
                        continue
                    if (interp := elf_interpreter(exe)) is not None:
                        found.append((exe, interp))
    return found


def check_bundled_binaries(c: Checker) -> None:
    c.section("venv binaries")
    binaries = bundled_elf_binaries()
    foreign = [(exe, interp) for exe, interp in binaries if not interp.startswith("/nix/store/")]
    if foreign:
        listing = ", ".join(f"{exe.name} -> {interp}" for exe, interp in foreign[:5])
        raise RuntimeError(
            f"{len(foreign)} of {len(binaries)} bundled ELF binarie(s) use a foreign "
            f"loader ({listing}); run `fix-cuda-binaries`"
        )
    c.info(f"{len(binaries)} bundled ELF executable(s), all with the NixOS loader")
    c.passed += 1


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #


def check_torch(c: Checker) -> None:
    c.section("pytorch")
    import torch

    c.info(
        f"version   {torch.__version__} (cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()})"
    )

    def available() -> str:
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        return torch.cuda.get_device_name(0)

    c.check("cuda available", available)
    if not torch.cuda.is_available():
        return

    def matmul_correct() -> str:
        torch.manual_seed(0)
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        expected = a @ b
        got = (a.cuda() @ b.cuda()).cpu()
        assert torch.allclose(got, expected, atol=1e-3), "gpu matmul differs from cpu"
        return "fp32 512x512 matches cpu"

    c.check("matmul fp32", matmul_correct)

    def matmul_bf16() -> str:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        out = (a @ b).float()
        assert torch.isfinite(out).all(), "non-finite values in bf16 matmul"
        return "bf16 2048x2048 finite"

    c.check("matmul bf16", matmul_bf16)

    def matmul_speed() -> str:
        size = 4096
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        for _ in range(2):
            a @ b
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10
        return f"fp16 4096x4096 {torch_tflops(size, elapsed):.0f} TFLOP/s"

    c.check("tensor cores", matmul_speed)

    def memory() -> str:
        free, total = torch.cuda.mem_get_info()
        return f"{total / 2**30:.1f} GiB device memory"

    c.check("device memory", memory)


# --------------------------------------------------------------------------- #
# JAX
# --------------------------------------------------------------------------- #


def check_jax(c: Checker) -> None:
    c.section("jax")
    import jax
    import jax.numpy as jnp

    c.info(f"version   jax {jax.__version__}, jaxlib {jax.lib.__version__}")
    c.info(f"backend   {jax.default_backend()}  {jax.devices()}")

    def backend_is_gpu() -> str:
        assert jax.default_backend() == "gpu", f"backend is {jax.default_backend()}"
        devices = jax.devices()
        assert devices and devices[0].platform == "gpu", f"no gpu device in {devices}"
        return str(devices[0])

    c.check("gpu backend", backend_is_gpu)
    if jax.default_backend() != "gpu":
        return

    def matmul_correct() -> str:
        a = jnp.arange(512 * 512, dtype=jnp.float32).reshape(512, 512) / 512
        expected = (a @ a)[0, :3]
        got = jax.jit(lambda x: x @ x)(a)[0, :3]
        got = jax.device_get(got)
        assert jnp.allclose(got, expected, rtol=1e-4), f"{got} != {expected}"
        return "jit matmul fp32 matches"

    c.check("jit matmul", matmul_correct)

    def bf16_matmul() -> str:
        a = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
        out = jax.device_get(jax.jit(lambda x: x @ x)(a))
        assert float(out[0, 0]) == 2048.0, f"expected 2048, got {out[0, 0]}"
        return "bf16 2048x2048 = 2048"

    c.check("matmul bf16", bf16_matmul)


# --------------------------------------------------------------------------- #
# cuTile
# --------------------------------------------------------------------------- #


def check_cutile(c: Checker) -> None:
    c.section("cutile")
    from importlib.metadata import version

    import torch

    if ct is None:
        raise RuntimeError(f"cuda.tile is not importable: {CUTILE_IMPORT_ERROR}")

    c.info(f"version   cuda-tile {version('cuda-tile')}, tileiras {version('nvidia-cuda-tileiras')}")

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda unavailable, cannot run cuTile kernels")
    torch.cuda.init()
    stream = torch.cuda.current_stream()
    n, tile = 4096, 1024

    @ct.kernel
    def vector_add(a, b, out, TILE: ct.Constant[int]):
        pid = ct.bid(0)
        at = ct.load(a, index=(pid,), shape=(TILE,))
        bt = ct.load(b, index=(pid,), shape=(TILE,))
        ct.store(out, index=(pid,), tile=at + bt)

    def launch_from_torch() -> str:
        a = torch.arange(n, dtype=torch.float32, device="cuda")
        b = torch.full((n,), 2.0, device="cuda")
        out = torch.zeros(n, dtype=torch.float32, device="cuda")
        grid = (ct.cdiv(n, tile), 1, 1)
        started = time.perf_counter()
        ct.launch(stream, grid, vector_add, (a, b, out, tile))
        torch.cuda.synchronize()
        assert torch.equal(out, a + b), "kernel output is wrong"
        return f"tileiras compile+kernel in {time.perf_counter() - started:.1f}s"

    c.check("torch tensors", launch_from_torch)

    def launch_from_jax() -> str:
        import jax
        import jax.numpy as jnp
        from cuda.tile.jax import OutputPlaceholder, cutile_call

        @jax.jit
        def add_two(x, y):
            grid = (ct.cdiv(x.shape[0], tile),)
            placeholder = OutputPlaceholder(x.shape, x.dtype)
            return cutile_call(grid, vector_add, (x, y, placeholder, tile))

        x = jnp.full((n,), 1.0, dtype=jnp.float32)
        y = jnp.full((n,), 2.0, dtype=jnp.float32)
        out = add_two(x, y)
        assert jnp.allclose(out, 3.0), f"unexpected result {out[:4]}"
        return "cuTile kernel inside jit"

    c.check("jax arrays (cutile_call)", launch_from_jax)


# --------------------------------------------------------------------------- #
# torch.compile (inductor + triton, needs triton's bundled ptxas to execute)
# --------------------------------------------------------------------------- #


def check_torch_compile(c: Checker) -> None:
    c.section("torch.compile (triton)")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda unavailable")

    def compiled_matmul() -> str:
        def fn(a, b):
            return (a @ b).relu() * 0.5

        torch.manual_seed(0)
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        expected = fn(a, b)
        started = time.perf_counter()
        compiled = torch.compile(fn, fullgraph=True)
        got = compiled(a, b)
        torch.cuda.synchronize()
        assert torch.allclose(got, expected, atol=1e-4), "compiled result differs"
        return f"compiled and ran in {time.perf_counter() - started:.0f}s"

    c.check("inductor kernel", compiled_matmul)


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run torch.compile / triton (slow, ~1 min)",
    )
    args = parser.parse_args()

    c = Checker()
    report_driver(c)

    for name, fn in (
        ("venv binaries", check_bundled_binaries),
        ("pytorch", check_torch),
        ("jax", check_jax),
        ("cutile", check_cutile),
        ("torch.compile", check_torch_compile if args.full else None),
    ):
        if fn is None:
            continue
        try:
            fn(c)
        except Exception as exc:  # noqa: BLE001 - report, keep going
            c.failures.append(name)
            print(f"  \033[31mFAIL\033[0m {name}: {type(exc).__name__}: {exc}")

    if c.failures:
        print(f"\n\033[31m{c.passed} passed, {len(c.failures)} failed\033[0m: {', '.join(c.failures)}")
        return 1
    print(f"\n\033[32mall {c.passed} checks passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
