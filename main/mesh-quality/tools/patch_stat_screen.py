#!/usr/bin/env python
"""Screen: do *local* patch statistics carry signal beyond the 27 global geometry
features? (cheap CPU HistGB, same folds as the shipped ablations)

Gate for the mesh-encoder investment: if the spatial heterogeneity of per-patch
statistics (min / p10 / p50 / p90 / max / std over the 256 patches) does not lift
the geometry and image+geometry baselines, a learned patch encoder has nothing to
build on.

    devenv shell -- uv run python main/mesh-quality/tools/patch_stat_screen.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import mesh_tokens as mt  # noqa: E402
from mesh_quality import metric, model  # noqa: E402


def patch_aggregates(stats: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per-stat order statistics over the valid patches (spatial heterogeneity)."""
    n, k, d = stats.shape
    feats = np.zeros((n, d * 6 + 1), dtype=np.float32)
    names: list[str] = []
    big = np.where(valid[..., None], stats, np.nan)
    q = np.nanpercentile(big, [10, 50, 90], axis=1)  # (3, n, d)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(big, axis=1)
        lo = np.nanmin(big, axis=1)
        hi = np.nanmax(big, axis=1)
    std = np.sqrt(np.nanmean((big - mean[:, None, :]) ** 2, axis=1))
    blocks = np.concatenate([lo, q[0], q[1], q[2], hi], axis=1)
    feats[:, : d * 5] = blocks
    feats[:, d * 5 : d * 6] = np.nan_to_num(std).reshape(n, d)
    feats[:, d * 6] = valid.mean(axis=1)
    names = [f"{s}_{t}" for t in ("min", "p10", "p50", "p90", "max") for s in mt.STAT_NAMES]
    names += [f"{s}_std" for s in mt.STAT_NAMES] + ["valid_frac"]
    return np.nan_to_num(feats), names


def image_summary(ds) -> np.ndarray:
    mean = np.asarray(ds.tile_mean, dtype=np.float32)
    mx = np.asarray(ds.tile_max, dtype=np.float32)
    grid = np.asarray(ds.tile_grid, dtype=np.float32).mean(axis=(2, 3))
    return np.concatenate([mean.reshape(len(ds), -1), mx.reshape(len(ds), -1), grid.reshape(len(ds), -1)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="s", help="DINOv3 cache key for the image baseline")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--split", default="train")
    ap.add_argument("--only", default=None, help="comma-separated variant names to run (default: all)")
    args = ap.parse_args()

    cache, data = TASK / "cache", TASK / "data"
    ds = model.load_dataset(cache, data, args.model, args.split)
    y = np.asarray(ds.labels)
    yq = metric.derive_quality(y)

    tk = mt.load_cache(cache / "mesh_tokens" / args.split)
    row = {item: i for i, item in enumerate(tk["item_ids"])}
    order = np.array([row[item] for item in ds.item_ids])
    stats = np.asarray(tk["stats"], dtype=np.float32)[order]
    valid = np.asarray(tk["valid"], dtype=bool)[order]
    glob = np.asarray(tk["globals"], dtype=np.float32)[order]
    print(f"{args.split}: {len(order)} items, valid patch fraction {valid.mean():.3f}, "
          f"items without valid patches {(valid.sum(1) == 0).sum()}")

    patch_feats, names = patch_aggregates(stats, valid)
    geom_z = ds.standardise(*model.geometry_stats(ds.geom))
    img = image_summary(ds)

    def zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)

    variants: dict[str, tuple[np.ndarray, int | None]] = {
        "geometry": (geom_z, None),
        "geometry+globals": (np.concatenate([geom_z, zscore(glob)], 1), None),
        "geometry+patchstats": (np.concatenate([geom_z, patch_feats], 1).astype(np.float32), None),
        "image+geometry": (np.concatenate([img, geom_z], 1).astype(np.float32), args.components),
        "image+geometry+patchstats": (np.concatenate([img, geom_z, patch_feats], 1).astype(np.float32), args.components),
    }
    rows = []
    for tag, (X, comps) in variants.items():
        if args.only and tag not in {v.strip() for v in args.only.split(",")}:
            continue
        t0 = time.time()
        oof = model.histgb_oof(X, y, k=args.folds, seed=args.seed, max_components=comps)
        _, res = metric.tune_thresholds(oof, y, yq, verbose=False)
        np.save(cache / f"oof_histgb_screen_{tag.replace('+', '_')}.npy", oof)
        rows.append({"variant": tag, "score": res["score"], "artefact": res["artefact_f1_weighted"],
                     "quality": res["quality_f1"], "secs": time.time() - t0, **res["per_label"]})
        print(f"{tag:28s} score={res['score']:6.3f} artefact={res['artefact_f1_weighted']:.4f} "
              f"quality={res['quality_f1']:.4f} ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(out[["score", "artefact", "quality", "noisy", "lowpoly", "abstract", "set", "simple", "artifacts", "open"]].round(3))
    out.to_csv(cache / "patch_stat_screen.csv")
    print(f"saved -> {cache / 'patch_stat_screen.csv'} (OOFs: oof_histgb_screen_*.npy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
