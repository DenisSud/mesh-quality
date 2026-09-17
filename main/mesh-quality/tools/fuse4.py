"""4-way fusion: fuse3 (s+b+L4) + the fine-tuned backbone member.

Same improved tuner as fuse3.py (fine grid + coordinate ascent + seeded anneal
for artefact F1w, decoupled quality OR-rule).  The ft member only has partial
OOF coverage tonight (fold 0, optionally fold 1), so:

  * weights are scanned on the rows where ALL members have OOF (the ft val folds);
  * the reported score is computed on that subset — compare it against the SAME
    subset scored with the shipped 3-way mix, not against full-OOF numbers;
  * a half-split check inside the subset guards threshold overfit.

    devenv shell -- uv run python main/mesh-quality/tools/fuse4.py \
        --ft-dir main/mesh-quality/cache/ft_full --folds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mesh_quality import metric  # noqa: E402
from fuse3 import (  # noqa: E402
    DEFECTS, artefact_f1w, dual_score, load_oof, quality_f1, tune_artefact,
    tune_quality,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--out-dir", type=Path, default=TASK / "cache" / "fuse4")
    ap.add_argument("--l4-dir", type=Path, default=TASK / "cache" / "l4_full")
    ap.add_argument("--ft-dir", type=Path, required=True)
    ap.add_argument("--folds", default="0", help="ft val fold indices, e.g. 0 or 0,1")
    ap.add_argument("--submission", type=Path, default=TASK / "submission.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache, out = args.cache_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ids = np.load(cache / "dinos_train" / "item_ids.npy")
    df = pd.read_csv(TASK / "data" / "train.csv").set_index("item_id").loc[list(ids)]
    y_all = df[list(DEFECTS)].to_numpy(np.int8)
    yq_all = df["quality"].to_numpy(np.int8)

    # rows where all four members have OOF = the union of ft val folds
    paths = []
    for f in args.folds.split(","):
        p = args.ft_dir / f"val_idx_{f}.npy"
        if not p.exists() and f == "0":
            p = args.ft_dir / "val_idx.npy"
        if not p.exists():
            raise SystemExit(f"missing {p} (train that ft fold first)")
        paths.append(p)
    keep = np.sort(np.concatenate([np.load(p) for p in paths]))
    y, yq = y_all[keep], yq_all[keep]
    oof = {
        "s": load_oof(cache, "oof_probe_s.npy")[keep],
        "b": load_oof(cache, "oof_probe_b.npy")[keep],
        "l4": load_oof(args.l4_dir, "oof_mesh.npy")[keep],
        "ft": np.load(args.ft_dir / "oof_ft.npy"),
    }
    if oof["ft"].shape[0] != len(keep):
        raise SystemExit(f"ft OOF rows {oof['ft'].shape[0]} != val rows {len(keep)}")
    print(f"[fuse4] tuning on {len(keep)} rows (ft val folds {args.folds})")

    # baseline on the SAME subset: shipped 3-way mix, coupled tuning
    cfg3 = json.loads((cache / "fuse3" / "fuse3.json").read_text())
    m3_sub = cfg3["weights"]["s"] * oof["s"] + cfg3["weights"]["b"] * oof["b"] + cfg3["weights"]["l4"] * oof["l4"]
    thr3 = np.asarray([cfg3["thresholds"][k] for k in DEFECTS], dtype=np.float32)
    qt3 = np.asarray([cfg3["quality_thresholds"][k] for k in DEFECTS], dtype=np.float32)
    s3, _, _ = dual_score(m3_sub, y, yq, thr3, qt3)
    print(f"[fuse4] shipped 3-way on subset : {s3:.4f}")

    results = []
    for ws in np.arange(0.0, 1.01, 0.1):
        for wb in np.arange(0.0, 1.01 - ws + 0.05, 0.1):
            for wl in np.arange(0.0, 1.01 - ws - wb + 0.05, 0.1):
                wft = round(1.0 - ws - wb - wl, 2)
                if wft < -1e-9:
                    continue
                m = ws * oof["s"] + wb * oof["b"] + wl * oof["l4"] + wft * oof["ft"]
                _, r = metric.tune_thresholds(m, y, yq)
                results.append((r["score"], round(ws, 2), round(wb, 2), round(wl, 2), wft))
    results.sort(reverse=True)
    print("top-5 weight combos (coupled ranking):")
    for s, ws, wb, wl, wft in results[:5]:
        print(f"  w=({ws:.2f}, {wb:.2f}, {wl:.2f}, {wft:.2f})  {s:.4f}")

    best = None
    for s_c, ws, wb, wl, wft in results[:3]:
        m = ws * oof["s"] + wb * oof["b"] + wl * oof["l4"] + wft * oof["ft"]
        thr, art = tune_artefact(m, y)
        qt, qual = tune_quality(m, yq)
        score, _, _ = dual_score(m, y, yq, thr, qt)
        print(f"improved w=({ws:.2f},{wb:.2f},{wl:.2f},{wft:.2f}) : {score:.4f}  art {art:.4f}  qual {qual:.4f}")
        if best is None or score > best[0]:
            best = (score, ws, wb, wl, wft, thr, qt, art, qual, m)
    score, ws, wb, wl, wft, thr, qt, art, qual, m = best
    print(f"\nBEST w=({ws},{wb},{wl},{wft})  subset OOF {score:.4f} (3-way on subset {s3:.4f})  "
          f"art {art:.4f}  qual {qual:.4f}")

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(y))
    ha, hb = perm[: len(y) // 2], perm[len(y) // 2:]
    thr_a, _ = tune_artefact(m[ha], y[ha], anneal=2000)
    qt_a, _ = tune_quality(m[ha], yq[ha])
    sb, _, _ = dual_score(m[hb], y[hb], yq[hb], thr_a, qt_a)
    print(f"half-split transfer (within subset): {sb:.4f}")

    blob = {
        "weights": {"s": float(ws), "b": float(wb), "l4": float(wl), "ft": float(wft)},
        "thresholds": {k: float(t) for k, t in zip(DEFECTS, thr)},
        "quality_thresholds": {k: float(t) for k, t in zip(DEFECTS, qt)},
        "subset_rows": int(len(keep)), "ft_folds": args.folds,
        "subset_score": float(score), "subset_3way_score": float(s3),
        "subset_artefact": float(art), "subset_quality": float(qual),
        "half_split": float(sb), "seed": 0,
    }
    (out / "fuse4_dry.json").write_text(json.dumps(blob, indent=2))
    print(f"[fuse4] wrote {out / 'fuse4_dry.json'}")
    if args.dry_run:
        return 0

    te_ids = np.load(cache / "dinos_test" / "item_ids.npy")
    te = {
        "s": load_oof(cache, "probs_test_s.npy"),
        "b": load_oof(cache, "probs_test_b.npy"),
        "l4": load_oof(args.l4_dir, "probs_test_l4.npy"),
        "ft": np.load(args.ft_dir / "probs_test_ft.npy"),
    }
    ft_te_ids = np.load(args.ft_dir / "item_ids_test.npy")
    if list(ft_te_ids) != list(te_ids):
        raise SystemExit("ft test row order differs from the cache order")
    probs_te = ws * te["s"] + wb * te["b"] + wl * te["l4"] + wft * te["ft"]
    d_te = (probs_te >= thr).astype(np.int8)
    q_te = ((probs_te >= qt).any(axis=1) == 0).astype(np.int8)
    metric.write_submission(args.submission, list(te_ids), d_te, q_te)
    blob["pred_positives"] = {k: int(v) for k, v in zip(DEFECTS, d_te.sum(axis=0))}
    blob["pred_quality_good"] = int(q_te.sum())
    blob["submission_md5"] = hashlib.md5(args.submission.read_bytes()).hexdigest()
    (out / "fuse4.json").write_text(json.dumps(blob, indent=2))
    np.save(out / "probs_test_fuse4.npy", probs_te)
    np.save(out / "oof_fuse4.npy", m)
    print(f"\nwrote {args.submission} (md5 {blob['submission_md5']})")
    print("  positives:", blob["pred_positives"], " quality-good:", blob["pred_quality_good"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
