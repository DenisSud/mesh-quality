#!/usr/bin/env python
"""Build the jupytext notebook sources into ``.ipynb`` and optionally smoke-run them.

    uv run python main/mesh-quality/tools/build_notebooks.py
    uv run python main/mesh-quality/tools/build_notebooks.py --run colab_solution
    uv run python main/mesh-quality/tools/build_notebooks.py --run colab_solution \\
        --include-slow --replace "EPOCHS, FOLDS = 40, 5" "EPOCHS, FOLDS = 2, 2"

Notebooks live as ``py:percent`` scripts (diffable, reviewable); the ``.ipynb``
files are generated.  Smoke runs execute the notebook with ``nbclient`` against a
kernelspec that points at the current interpreter, skipping cells tagged ``skip``
(Colab-only installs) and -- unless ``--include-slow`` -- cells tagged ``slow``.
``--replace OLD NEW`` rewrites the source text before execution, which is how the
smoke test shortens the training cell without touching the delivered notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
SRC = TASK / "notebooks"
KERNEL = "aiijc-uv"


def ensure_kernel() -> str:
    """Register a kernelspec for the running interpreter (idempotent)."""
    spec_dir = Path(sys.prefix) / "share" / "jupyter" / "kernels" / KERNEL
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "aiijc (uv venv)",
                "language": "python",
            },
            indent=1,
        )
    )
    return KERNEL


def build(name: str, replacements: list[list[str]], run: bool, include_slow: bool, timeout: int) -> Path:
    import jupytext
    import nbformat

    src = SRC / f"{name}.py"
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    text_clean = src.read_text()

    nb = jupytext.reads(text_clean, fmt="py:percent")
    out = SRC / f"{name}.ipynb"
    jupytext.write(nb, out)
    print(f"[notebook] {out.relative_to(TASK)}  ({len(nb.cells)} cells)")
    if not run:
        return out

    # Подмены (локальные пути, отладочные лимиты) применяются только к прогону,
    # чтобы в отгружаемый .ipynb не попали ссылки smoke-теста.
    text_run = text_clean
    for old, new in replacements:
        if old not in text_run:
            raise SystemExit(f"--replace: {old!r} not found in {src.name}")
        text_run = text_run.replace(old, new)
    nb = jupytext.reads(text_run, fmt="py:percent")

    kept = []
    for cell in nb.cells:
        tags = list(cell.metadata.get("tags", []))
        if "skip" in tags:
            continue
        if "slow" in tags and not include_slow:
            continue
        cell.metadata["tags"] = [t for t in tags if t not in {"skip", "slow"}]
        kept.append(cell)
    print(f"[notebook] executing {len(kept)}/{len(nb.cells)} cells "
          f"({'slow cells included' if include_slow else 'slow cells skipped'})")

    from nbclient import NotebookClient

    executed = nbformat.v4.new_notebook(cells=kept, metadata=nb.metadata)
    client = NotebookClient(
        executed,
        timeout=timeout,
        kernel_name=ensure_kernel(),
        allow_errors=False,
        resources={"metadata": {"path": str(TASK)}},
    )
    client.execute()
    out_exec = SRC / f"{name}.executed.ipynb"
    nbformat.write(executed, out_exec)
    print(f"[notebook] smoke run ok -> {out_exec.relative_to(TASK)}")
    return out_exec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="execute this notebook after building it")
    ap.add_argument("--include-slow", action="store_true", help="also execute cells tagged 'slow'")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    ap.add_argument(
        "--replace",
        nargs=2,
        action="append",
        default=[],
        metavar=("OLD", "NEW"),
        help="textual substitution applied before execution (repeatable)",
    )
    args = ap.parse_args(argv)

    names = sorted(p.stem for p in SRC.glob("*.py") if not p.name.startswith("_"))
    for name in names:
        run = args.run == name
        build(name, args.replace if run else [], run=run, include_slow=args.include_slow, timeout=args.timeout)
    if args.run and args.run not in names:
        raise SystemExit(f"--run {args.run}: no notebooks/{args.run}.py (have: {names})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
