#!/usr/bin/env python
"""Hyperparameter ablations for the attentive probe (presentation / final-stage log).

Every run uses the *same* recipe as the shipped probe -- 5 folds, 40 epochs,
AdamW lr 1e-3, batch 128, BCE(pos_weight=(neg/pos)^0.5) -- and changes exactly one
thing.  The fold split is always ``folds(seed=0)``; ``--model-seed`` varies only
the initialisation, so seed variance is separated from split variance.

Outputs (nothing touches the shipped artifacts):

    cache/ablations/<tag>.json     score, per-label F1, config
    cache/ablations/<tag>.npy      the OOF probability matrix
    cache/ablations/curves.json    per-epoch train/val loss of one fold

Usage (from the repo root):

    devenv shell -- uv run python main/mesh-quality/tools/hyperparams.py --list
    devenv shell -- uv run python main/mesh-quality/tools/hyperparams.py curves
    devenv shell -- uv run python main/mesh-quality/tools/hyperparams.py depth2 depth6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import images, metric, model  # noqa: E402

OUT = TASK / "cache" / "ablations"
#: Training knobs passed through to `model.run_cv` / `model.train_model`.
TRAIN_KEYS = ("epochs", "batch", "lr", "wd", "pos_weight_pow")
# Main-stage shipped recipe (40 epochs) -- kept for the log / reruns of old tags.
OLD = {key: getattr(model.TrainRecipe, key) for key in TRAIN_KEYS}
OLD |= {"depth": model.ProbeConfig.depth, "cache": "cache"}
# Final-stage default: 10 epochs (validated: 13.47 vs 13.45 at 40, within noise).
BASE = {**OLD, "epochs": 10}


def run_specs() -> dict[str, dict]:
    """tag -> overrides on top of BASE (cache is relative to the task dir)."""
    return {
        # anchor: the shipped recipe (8x8 pooled grid + tile summaries) at 10 epochs
        "g8summ10": {**BASE},
        # --- main-stage ablations (40 epochs, shipped recipe), kept for the log ---
        "curves":   {**OLD},  # special: one fold, per-epoch train/val loss
        "epochs10": {**OLD, "epochs": 10},
        "depth2":   {**OLD, "depth": 2},
        "depth6":   {**OLD, "depth": 6},
        "posw025":  {**OLD, "pos_weight_pow": 0.25},
        "posw075":  {**OLD, "pos_weight_pow": 0.75},
        "seed1":    {**OLD, "model_seed": 1},
        "seed2":    {**OLD, "model_seed": 2},
    }


def _load(spec: dict, split: str = "train"):
    cache_dir = TASK / spec["cache"]
    return model.load_dataset(cache_dir, TASK / "data", "s", split)


def _geometry(ds: model.Dataset):
    median, scale = model.geometry_stats(ds.geom)
    return ds.standardise(median, scale), median, scale


def _cfg(ds: model.Dataset, spec: dict) -> model.ProbeConfig:
    return model.ProbeConfig(dim=ds.dim, n_geom=ds.geom.shape[1], grid=ds.grid, depth=spec["depth"])


def _train_kwargs(spec: dict, epochs: int | None = None) -> dict:
    kwargs = {key: spec[key] for key in TRAIN_KEYS}
    if epochs is not None:
        kwargs["epochs"] = epochs
    return kwargs


def run_one(tag: str, spec: dict, epochs: int | None = None, only_fold: int | None = None) -> None:
    (OUT).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if spec["cache"] != "cache" and not (TASK / spec["cache"] / "dinos_train" / "meta.json").exists():
        print(f"[{tag}] extracting native-grid features into {spec['cache']} (this takes hours)", flush=True)
        images.extract_split(TASK / "data", "train", TASK / spec["cache"], model_key="s")
        for name in ("geometry_train.csv", "geometry_test.csv"):
            src = TASK / "cache" / name
            if src.exists() and not (TASK / spec["cache"] / name).exists():
                (TASK / spec["cache"] / name).write_bytes(src.read_bytes())

    if tag == "curves":
        ds = _load(spec)
        gz, *_ = _geometry(ds)
        cfg = _cfg(ds, spec)
        val = model.folds(len(ds), 5, seed=0)[0]
        train = np.setdiff1d(np.arange(len(ds)), val)
        print(f"[{tag}] one fold, per-epoch train/val loss", flush=True)
        val_history: list[float] = []
        _, hist = model.train_model(
            ds, cfg, gz, train, val_idx=val, val_history=val_history, log_every=0,
            **_train_kwargs(spec),
        )
        (OUT / "curves.json").write_text(json.dumps({
            "train_loss": hist, "val_loss": val_history, "epochs": spec["epochs"],
            "fold": 0, "config": {k: v for k, v in spec.items()},
        }, indent=2))
        print(f"[{tag}] train {hist[0]:.4f} -> {hist[-1]:.4f}, val {val_history[-1]:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        return

    ds = _load(spec)
    gz, *_ = _geometry(ds)
    cfg = _cfg(ds, spec)
    labels = pd.read_csv(TASK / "data" / "train.csv").set_index("item_id")
    ids = np.asarray(ds.item_ids)
    y = labels.loc[list(ids), list(metric.DEFECTS)].to_numpy(dtype=np.int8)
    yq = metric.derive_quality(y)

    print(f"[{tag}] {len(ds)} items, depth {cfg.depth}, pos_pow {spec['pos_weight_pow']}, "
          f"model_seed {spec.get('model_seed', 0)}"
          + (f", SCREEN fold {only_fold} epochs {epochs}" if only_fold is not None or epochs is not None else ""),
          flush=True)
    oof, done = model.run_cv(
        ds, cfg, gz, fold_seed=0, model_seed=int(spec.get("model_seed", 0)),
        only_folds=None if only_fold is None else [only_fold], **_train_kwargs(spec, epochs),
    )
    tag_out = tag if done.all() else f"{tag}_screen"
    thresholds, res = metric.tune_thresholds(oof[done], y[done], yq[done], verbose=False)
    print(f"[{tag}] score {res['score']:.3f}/20  artefact {res['artefact_f1_weighted']:.4f}  "
          f"quality {res['quality_f1']:.4f}  on {int(done.sum())} items  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.save(OUT / f"{tag_out}.npy", oof)
    (OUT / f"{tag_out}.json").write_text(json.dumps({
        "tag": tag, "screen": not done.all(), "n_scored": int(done.sum()),
        "epochs": epochs, "only_fold": only_fold,
        "score": res["score"], "artefact_f1_weighted": res["artefact_f1_weighted"],
        "quality_f1": res["quality_f1"], "per_label": res["per_label"], "support": res["support"],
        "pred_positives": res["pred_positives"], "thresholds": [float(t) for t in thresholds],
        "config": {k: v for k, v in spec.items()}, "n": len(ds), "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    spec = run_specs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help=f"one of: {', '.join(spec)}")
    ap.add_argument("--list", action="store_true", help="print the run table and exit")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (screening)")
    ap.add_argument("--fold", type=int, default=None, help="run a single fold only (screening)")
    args = ap.parse_args(argv)
    if args.list or not args.runs:
        for tag, s in spec.items():
            diff = {k: v for k, v in s.items() if BASE.get(k) != v}
            print(f"  {tag:<8} {s['cache']:<12} depth {s['depth']}  "
                  f"pos_pow {s['pos_weight_pow']}  seed {s.get('model_seed', 0)}  {diff or '(anchor recipe)'}")
        return 0
    unknown = [r for r in args.runs if r not in spec]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; have {list(spec)}")
    for tag in args.runs:
        run_one(tag, spec[tag], epochs=args.epochs, only_fold=args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
