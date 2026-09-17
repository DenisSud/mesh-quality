"""Compare two OOF probability matrices on the rows both cover.

Single-fold, tuned-threshold scores swing by >0.5 on batch size alone, so a raw
score difference says very little. This tool pairs the two runs on the shared
rows, prints per-label F1 side by side, and bootstraps the *score difference*:

    devenv shell -- uv run python main/mesh-quality/tools/compare_oof.py \
        --a cache/oof_probe_s.npy --b cache/l4_full/oof_mesh.npy \
        --b-partial cache/l4_full/oof_partial.npz

``--b-partial`` restricts the comparison to the rows ``b`` actually scored (its
``done`` mask), which is what makes a 2-fold run comparable to the 5-fold
reference: the reference is cut down to the same rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import images, metric  # noqa: E402


def labels_for(item_ids: np.ndarray, data_dir: Path) -> np.ndarray:
    lab = pd.read_csv(data_dir / "train.csv").set_index("item_id").reindex(list(item_ids))
    return lab[list(metric.DEFECTS)].to_numpy(np.int8)


def score(mask: np.ndarray, oof: np.ndarray, y: np.ndarray, yq: np.ndarray) -> dict:
    return metric.tune_thresholds(oof[mask], y[mask], yq[mask], verbose=False)[1]


def bootstrap_delta(
    mask: np.ndarray, a: np.ndarray, b: np.ndarray, y: np.ndarray, yq: np.ndarray, n_boot: int, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(mask)
    deltas = np.empty(n_boot, np.float64)
    for i in range(n_boot):
        pick = rng.choice(rows, size=len(rows), replace=True)
        sub = np.zeros_like(mask)
        sub[pick] = True
        deltas[i] = score(sub, b, y, yq)["score"] - score(sub, a, y, yq)["score"]
    return float(np.mean(deltas)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference OOF (npy)")
    ap.add_argument("--b", type=Path, required=True, help="candidate OOF (npy)")
    ap.add_argument("--b-partial", type=Path, default=None, help="oof_partial.npz of the candidate run")
    ap.add_argument("--b-folds", default=None, help="fold indices to restrict to, e.g. 0,1 (same folds as run_cv)")
    ap.add_argument("--folds-seed", type=int, default=None, help="fold seed used by the candidate run")
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--data-dir", type=Path, default=TASK / "data")
    ap.add_argument("--model", default="s", help="DINOv3 cache key (row order source)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tag-a", default=None)
    ap.add_argument("--tag-b", default=None)
    args = ap.parse_args()

    ids = np.load(images.cache_path(args.cache_dir, args.model, "train") / "item_ids.npy")
    y = labels_for(ids, args.data_dir)
    yq = metric.derive_quality(y)
    a, b = np.load(args.a), np.load(args.b)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.ones(len(a), bool)
    if args.b_folds:
        from mesh_quality import model as M

        seed = M.SEED if args.folds_seed is None else args.folds_seed
        split = M.folds(len(a), 5, seed)
        mask[:] = False
        for f in (int(v) for v in args.b_folds.split(",")):
            mask[split[f]] = True
    if args.b_partial is not None:
        done = np.load(args.b_partial)["done"]
        if len(done) != len(mask):
            raise SystemExit(f"{args.b_partial}: done has {len(done)} rows, expected {len(mask)}")
        mask &= done
    tag_a = args.tag_a or args.a.stem
    tag_b = args.tag_b or args.b.stem

    ra, rb = score(mask, a, y, yq), score(mask, b, y, yq)
    print(f"rows compared: {int(mask.sum())} / {len(mask)}   ({args.b_partial.name if args.b_partial else 'all rows'})")
    print(f"{'':16s} {'score':>8s} {'artefact':>9s} {'quality':>8s}")
    for tag, r in ((tag_a, ra), (tag_b, rb)):
        print(f"{tag:16s} {r['score']:8.3f} {r['artefact_f1_weighted']:9.3f} {r['quality_f1']:8.3f}")
    print(f"{'delta':16s} {rb['score'] - ra['score']:8.3f} {rb['artefact_f1_weighted'] - ra['artefact_f1_weighted']:9.3f} "
          f"{rb['quality_f1'] - ra['quality_f1']:8.3f}")

    print(f"\n{'label':14s} {'support':>8s} {tag_a[:9]:>9s} {tag_b[:9]:>9s} {'delta':>7s}")
    for label in metric.DEFECTS:
        support = int(y[mask][:, list(metric.DEFECTS).index(label)].sum())
        print(f"{label:14s} {support:8d} {ra['per_label'][label]:9.3f} {rb['per_label'][label]:9.3f} "
              f"{rb['per_label'][label] - ra['per_label'][label]:7.3f}")

    if args.boot:
        mean, lo, hi = bootstrap_delta(mask, a, b, y, yq, args.boot)
        verdict = "B better" if lo > 0 else ("A better" if hi < 0 else "indistinguishable")
        print(f"\nbootstrap score difference ({tag_b} - {tag_a}), {args.boot} resamples: "
              f"{mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
