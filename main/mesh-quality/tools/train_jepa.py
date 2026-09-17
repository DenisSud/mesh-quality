#!/usr/bin/env python
"""Train the mesh-patch JEPA encoder (A) and dump frozen embeddings for the probe (C).

    devenv shell -- uv run python main/mesh-quality/tools/train_jepa.py --epochs 500 --batch 256
    devenv shell -- uv run python main/mesh-quality/tools/train_jepa.py --dump --ckpt ...

Labels are only used for the *monitoring* probe (5-fold logistic on the pooled
embedding) - the encoder itself is unsupervised.
"""

from __future__ import annotations

import os

# must be set before torch initialises the CUDA allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import jepa, metric  # noqa: E402


def load_labels(data_dir: Path, item_ids: list[str]) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(data_dir / "train.csv").set_index("item_id")
    missing = [i for i in item_ids if i not in frame.index]
    if missing:
        raise SystemExit(f"{len(missing)} cache items missing from train.csv (e.g. {missing[0]})")
    return frame.loc[item_ids, list(metric.DEFECTS)].to_numpy(dtype=np.int8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache" / "mesh_tokens")
    ap.add_argument("--data-dir", type=Path, default=TASK / "data")
    ap.add_argument("--out", type=Path, default=TASK / "cache" / "jepa_pretrain" / "run1")
    ap.add_argument("--split", default="train", choices=["train", "test", "both"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--pred-d", type=int, default=192)
    ap.add_argument("--pred-depth", type=int, default=4)
    ap.add_argument("--mask-lo", type=float, default=0.15)
    ap.add_argument("--mask-hi", type=float, default=0.40)
    ap.add_argument("--mask-prob", type=float, default=0.75)
    ap.add_argument("--stats-weight", type=float, default=0.10)
    ap.add_argument("--sigreg-weight", type=float, default=0.05)
    ap.add_argument("--point-dropout", type=float, default=0.20)
    ap.add_argument("--patch-dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", action="store_true", help="dump frozen embeddings instead of training")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--pool-only", action="store_true", help="dump only the pooled embedding")
    args = ap.parse_args(argv)

    cfg = jepa.JepaConfig(
        d=args.d, depth=args.depth, heads=args.heads, pred_d=args.pred_d, pred_depth=args.pred_depth,
        mask_lo=args.mask_lo, mask_hi=args.mask_hi, mask_probability=args.mask_prob,
        stats_weight=args.stats_weight, sigreg_weight=args.sigreg_weight,
        point_dropout=args.point_dropout, patch_dropout=args.patch_dropout,
        lr=args.lr, wd=args.wd, batch=args.batch, epochs=args.epochs, amp=args.amp, workers=args.workers,
    )
    if args.dump:
        ckpt = args.ckpt or (args.out / "ckpt.pt")
        splits = ["test"] if args.split == "test" else ["train", "test"]
        for split in splits:
            stem = args.out / f"{split}"
            t0 = time.time()
            jepa.dump_embeddings(ckpt, args.cache_dir, split, stem, device=args.device or "cuda",
                                 pool_only=args.pool_only)
            print(f"dumped {split} -> {stem}_*.npy ({time.time() - t0:.0f}s)", flush=True)
        return 0

    labels, label_rows = None, None
    if args.split in ("train", "both"):
        n_train = len(jepa.MeshPatchDataset(args.cache_dir, "train"))
        ids = jepa.make_dataset(args.cache_dir, args.split).item_ids[:n_train]
        labels = load_labels(args.data_dir, ids)
        label_rows = n_train
    print(f"training {args.epochs} epochs, config {cfg}", flush=True)
    path = jepa.train(
        args.cache_dir, args.out, cfg, split=args.split, labels=labels, label_rows=label_rows,
        log_every=args.log_every, eval_every=args.eval_every, save_every=args.save_every,
        device=args.device, seed=args.seed, max_minutes=args.max_minutes,
    )
    print(f"checkpoint -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
