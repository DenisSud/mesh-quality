#!/usr/bin/env python
"""Shallow-model ablations on the *same* frozen features (CPU only).

Answers two questions the presentation asks:

* how much do the image features add over the 27 geometry features alone;
* does the attentive probe actually beat a HistGradientBoosting on a PCA of the
  same pooled image features (i.e. is the transformer earning its parameters).

Writes ``cache/oof_histgb_<tag>.npy`` for each variant.  Runs on CPU: no GPU
needed, safe to launch while training occupies the card.

    uv run python main/mesh-quality/tools/ablation_histgb.py [--folds 5] [--seed 42]
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

from mesh_quality import metric, model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="s", help="DINOv3 cache key")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--components", type=int, default=256, help="PCA components for image features")
    args = ap.parse_args(argv)

    cache, data = TASK / "cache", TASK / "data"
    ds = model.load_dataset(cache, data, args.model, "train")
    y = ds.labels
    assert y is not None
    yq = metric.derive_quality(y)
    geom_z = ds.standardise(*model.geometry_stats(ds.geom))

    def image_summary() -> np.ndarray:
        """Flattened per-item summary of the cached image features."""
        mean = np.asarray(ds.tile_mean, dtype=np.float32)
        mx = np.asarray(ds.tile_max, dtype=np.float32)
        grid = np.asarray(ds.tile_grid, dtype=np.float32).mean(axis=(2, 3))
        return np.concatenate(
            [mean.reshape(len(ds), -1), mx.reshape(len(ds), -1), grid.reshape(len(ds), -1)], axis=1
        )

    variants = {
        "geometry": (geom_z, None),
        "image": (image_summary(), args.components),
        "image+geometry": (
            np.concatenate([image_summary(), geom_z], axis=1).astype(np.float32),
            args.components,
        ),
    }
    summary = {}
    for tag, (X, comps) in variants.items():
        t0 = time.time()
        oof = model.histgb_oof(X, y, k=args.folds, seed=args.seed, max_components=comps)
        thr, res = metric.tune_thresholds(oof, y, yq, verbose=False)
        np.save(cache / f"oof_histgb_{tag.replace('+', '_')}.npy", oof)
        summary[tag] = res
        print(
            f"[ablation] {tag:<15} X={np.shape(X)}  score {res['score']:6.3f}  "
            f"qualityF1 {res['quality_f1']:.4f}  artF1w {res['artefact_f1_weighted']:.4f}  "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )
        print("            per-label " + " ".join(f"{res['per_label'][n]:.2f}" for n in metric.DEFECTS), flush=True)

    table = pd.DataFrame(
        {
            tag: {
                "score": round(res["score"], 3),
                "quality_f1": round(res["quality_f1"], 4),
                "artefact_f1_weighted": round(res["artefact_f1_weighted"], 4),
            }
            for tag, res in summary.items()
        }
    ).T
    print("\n" + table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
