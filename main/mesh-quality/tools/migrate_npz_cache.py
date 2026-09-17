#!/usr/bin/env python
"""Convert a legacy ``.npz`` feature cache into the current directory format.

Main-stage caches were one ``dino{key}_{split}.npz`` per backbone x split; the
code now reads a directory per cache (``images.load_cache``).  Arrays are copied
verbatim (all fp16), so the converted cache is bit-identical to the archive --
that is what makes shipped checkpoints/thresholds reproducible against it.

    uv run python main/mesh-quality/tools/migrate_npz_cache.py \\
        main/mesh-quality/cache/backup_g8v1/dinos_train.npz --out main/mesh-quality/cache/legacy_g8

A converted cache is a normal cache: point ``--out`` at ``cache/`` itself to use
it directly (it will refuse to overwrite an existing non-empty cache directory).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import images  # noqa: E402


def migrate(npz_path: Path, out_dir: Path) -> Path:
    with np.load(npz_path) as z:
        grid = int(z["grid_size"])
        dim = int(z["dim"])
        item_ids = np.asarray(z["item_ids"])
        arrays = {name: np.asarray(z[name]) for name in ("tile_mean", "tile_max", "tile_grid")}
    if grid != images.POOLED_GRID:
        raise SystemExit(f"{npz_path}: grid_size {grid}, expected {images.POOLED_GRID} (pooled cache)")
    model_key, _, split = npz_path.stem[len("dino") :].partition("_")
    if model_key not in images.MODELS or split not in ("train", "test"):
        raise SystemExit(f"{npz_path}: cannot parse backbone/split from the file name")
    dest = images.cache_path(out_dir, model_key, split)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} already exists and is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(dest / f"{name}.npy", arr)
    np.save(dest / "item_ids.npy", item_ids)
    meta = {
        "model_key": model_key,
        "model": images.MODELS[model_key],
        "split": split,
        "grid_size": grid,
        "native_grid_size": images.GRID,
        "dim": dim,
        "n_tiles": images.N_TILES,
        "n_items": len(item_ids),
        "dtype": "float16",
        "converted_from": str(npz_path),
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"[migrate] {npz_path} -> {dest} ({len(item_ids)} items, grid {grid}, dim {dim})")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="legacy cache, e.g. cache/backup_g8v1/dinos_train.npz")
    ap.add_argument("--out", type=Path, default=TASK / "cache" / "legacy_g8", help="output cache dir")
    args = ap.parse_args(argv)
    migrate(args.npz, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
