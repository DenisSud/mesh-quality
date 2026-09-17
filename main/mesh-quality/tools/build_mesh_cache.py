#!/usr/bin/env python
"""Build the mesh-token cache for one split (resumable, size-aware scheduling).

    devenv shell -- uv run python main/mesh-quality/tools/build_mesh_cache.py --split all

Small meshes go through the worker pool; the few huge meshes (>= 30 MB npz) are
tokenized serially afterwards to keep peak RAM bounded.
"""

from __future__ import annotations

import os

# single-threaded workers: the pool is CPU-bound numpy, and BLAS thread pools
# oversubscribe the machine badly (6 workers x 12 threads -> 10x slowdown)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR / "src"))

from mesh_quality import mesh_tokens as mt  # noqa: E402

BIG_BYTES = 30_000_000


def build_split(split: str, data_dir: Path, cache_dir: Path, workers: int, limit: int | None) -> None:
    paths = sorted((data_dir / split).glob("*.npz"))
    if limit:
        paths = paths[:limit]
    sizes = np.array([p.stat().st_size for p in paths])
    small = np.flatnonzero(sizes < BIG_BYTES).tolist()
    big = sorted(np.flatnonzero(sizes >= BIG_BYTES).tolist(), key=lambda i: sizes[i])
    out = cache_dir / split
    print(f"[{split}] {len(paths)} meshes: {len(small)} small + {len(big)} big -> {out}", flush=True)
    t0 = time.perf_counter()
    mt.build_cache(paths, out, workers=workers, rows=small)
    print(f"[{split}] small done in {time.perf_counter() - t0:.1f}s", flush=True)
    if big:
        t1 = time.perf_counter()
        mt.build_cache(paths, out, workers=1, rows=big, resume=True)
        print(f"[{split}] big done in {time.perf_counter() - t1:.1f}s", flush=True)
    total = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
    print(f"[{split}] cache {total:.2f} GB in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def validate(split: str, cache_dir: Path) -> None:
    cache = mt.load_cache(cache_dir / split)
    stats, valid, glob = cache["stats"], cache["valid"], cache["globals"]
    print(f"[{split}] rows={len(cache['item_ids'])} valid patches={valid.mean():.3f} "
          f"finite={np.isfinite(np.asarray(stats, dtype=np.float32)).mean():.4f}")
    print(f"  has_area={float(np.mean(glob[:, 10])):.4f} has_structure={float(np.mean(glob[:, 9])):.4f} "
          f"log_diag p1/p50/p99={np.round(np.percentile(glob[:, 0], [1, 50, 99]), 2)}")
    print(f"  stats p1/p50/p99 per column (clipped at +-30):")
    for j, name in enumerate(mt.STAT_NAMES):
        col = np.asarray(stats[:, :, j], dtype=np.float32).ravel()
        q = np.percentile(col, [1, 50, 99])
        print(f"    {name:22s} {q[0]:9.3f} {q[1]:9.3f} {q[2]:9.3f}  zeros={np.mean(col == 0):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--data-dir", type=Path, default=TASK_DIR / "data")
    ap.add_argument("--cache-dir", type=Path, default=TASK_DIR / "cache" / "mesh_tokens")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N items")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        if not args.validate_only:
            build_split(split, args.data_dir, args.cache_dir, args.workers, args.limit)
        validate(split, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
