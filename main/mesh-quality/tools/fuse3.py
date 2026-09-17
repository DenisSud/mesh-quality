"""3-way fusion (DINOv3-s + DINOv3-b + L4 joint mesh probe) with an improved
threshold tuner, and submission writing.

Improvements over the shipped ``fuse`` path (all measured on cached OOF,
/tmp measurements 17 Sep):
  * finer threshold grid (0.02 vs 0.05) + annealing refinement;
  * decoupled quality rule: quality keeps its own per-label thresholds
    (quality_hat = 1 iff all p_i < qt_i) instead of the all-zero identity.
    The submission format allows any quality column; the two metric terms
    are scored independently, so decoupling is strictly richer.
  * half-split transfer check: tune on one half of the OOF, score on the
    other, so the reported gain is not pure threshold overfit.

    devenv shell -- uv run python main/mesh-quality/tools/fuse3.py --dry-run
    devenv shell -- uv run python main/mesh-quality/tools/fuse3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import metric  # noqa: E402

DEFECTS = metric.DEFECTS
FINE = np.round(np.arange(0.02, 0.981, 0.02), 3)


# --------------------------------------------------------------------------- #
# improved threshold tuning
# --------------------------------------------------------------------------- #

def artefact_f1w(y, d):
    return float(f1_score(y, d, average="weighted", zero_division=0))


def quality_f1(yq, qp):
    return float(f1_score(yq, qp, pos_label=1, zero_division=0))


def tune_artefact(probs, y, thr0=None, passes=6, anneal=4000, seed=0):
    """Fine-grid coordinate ascent on F1_weighted + seeded anneal refinement."""
    thr = np.full(probs.shape[1], 0.5) if thr0 is None else np.asarray(thr0, float).copy()
    best = artefact_f1w(y, (probs >= thr).astype(np.int8))
    for _ in range(passes):
        improved = False
        for k in range(probs.shape[1]):
            keep, keep_s = thr[k], best
            for v in FINE:
                cand = thr.copy(); cand[k] = v
                s = artefact_f1w(y, (probs >= cand).astype(np.int8))
                if s > keep_s + 1e-12:
                    keep, keep_s = v, s
            if keep != thr[k]:
                thr[k] = keep; best = keep_s; improved = True
        if not improved:
            break
    rng = np.random.default_rng(seed)
    cur, cur_s = thr.copy(), best
    for _ in range(anneal):
        cand = cur.copy()
        k = int(rng.integers(len(cur)))
        cand[k] = float(np.clip(cur[k] + rng.normal(0, 0.03), 0.01, 0.99))
        s = artefact_f1w(y, (probs >= cand).astype(np.int8))
        if s > cur_s:
            cur, cur_s = cand, s
    return (cur if cur_s > best else thr), max(cur_s, best)


def tune_quality(probs, yq, passes=6):
    """OR-rule thresholds: quality_hat = 1 iff all p_i < qt_i."""
    qt = np.full(probs.shape[1], 0.5)

    def pred(t):
        return ((probs >= t).any(axis=1) == 0).astype(np.int8)

    best = quality_f1(yq, pred(qt))
    for _ in range(passes):
        improved = False
        for k in range(probs.shape[1]):
            keep, keep_s = qt[k], best
            for v in FINE:
                cand = qt.copy(); cand[k] = v
                s = quality_f1(yq, pred(cand))
                if s > keep_s + 1e-12:
                    keep, keep_s = v, s
            if keep != qt[k]:
                qt[k] = keep; best = keep_s; improved = True
        if not improved:
            break
    return qt, best


def dual_score(probs, y, yq, thr, qt):
    d = (probs >= thr).astype(np.int8)
    qp = ((probs >= qt).any(axis=1) == 0).astype(np.int8)
    return 10.0 * artefact_f1w(y, d) + 10.0 * quality_f1(yq, qp), d, qp


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_oof(cache: Path, name: str) -> np.ndarray:
    p = cache / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return np.load(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--out-dir", type=Path, default=TASK / "cache" / "fuse3")
    ap.add_argument("--l4-dir", type=Path, default=TASK / "cache" / "l4_full")
    ap.add_argument("--dry-run", action="store_true", help="OOF analysis only, no submission")
    ap.add_argument("--submission", type=Path, default=TASK / "submission.csv")
    args = ap.parse_args()

    cache, out = args.cache_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ids = np.load(cache / "dinos_train" / "item_ids.npy")
    df = pd.read_csv(TASK / "data" / "train.csv").set_index("item_id").loc[list(ids)]
    y = df[list(DEFECTS)].to_numpy(np.int8)
    yq = df["quality"].to_numpy(np.int8)

    oof = {
        "s": load_oof(cache, "oof_probe_s.npy"),
        "b": load_oof(cache, "oof_probe_b.npy"),
        "l4": load_oof(args.l4_dir, "oof_mesh.npy"),
    }
    n = len(y)
    for k, v in oof.items():
        if v.shape != (n, len(DEFECTS)):
            raise SystemExit(f"oof[{k}] shape {v.shape} != {(n, len(DEFECTS))} (finish folds 2-4 first?)")

    # ---- shipped baseline (coupled, coarse) ----
    _, r0 = metric.tune_thresholds(0.5 * (oof["s"] + oof["b"]), y, yq)
    print(f"shipped s+b coupled           : {r0['score']:.4f}  art {r0['artefact_f1_weighted']:.4f}  qual {r0['quality_f1']:.4f}")

    # ---- weight scan (coarse coupled tuner ranks the weights) ----
    results = []
    for ws in np.arange(0.0, 1.01, 0.1):
        for wb in np.arange(0.0, 1.01 - ws + 0.05, 0.1):
            wl = round(1.0 - ws - wb, 2)
            if wl < -1e-9:
                continue
            m = ws * oof["s"] + wb * oof["b"] + wl * oof["l4"]
            _, r = metric.tune_thresholds(m, y, yq)
            results.append((r["score"], round(ws, 2), round(wb, 2), wl))
    results.sort(reverse=True)
    print("top-5 weight combos (coupled ranking):")
    for s, ws, wb, wl in results[:5]:
        print(f"  w=({ws:.2f}, {wb:.2f}, {wl:.2f})  {s:.4f}")

    # ---- improved tuning on the top-3 combos ----
    best = None
    for s_c, ws, wb, wl in results[:3]:
        m = ws * oof["s"] + wb * oof["b"] + wl * oof["l4"]
        thr, art = tune_artefact(m, y)
        qt, qual = tune_quality(m, yq)
        score, d, qp = dual_score(m, y, yq, thr, qt)
        print(f"improved w=({ws:.2f},{wb:.2f},{wl:.2f}) : {score:.4f}  art {art:.4f}  qual {qual:.4f}")
        if best is None or score > best[0]:
            best = (score, ws, wb, wl, thr, qt, art, qual, m)
    score, ws, wb, wl, thr, qt, art, qual, m = best
    per_label = f1_score(y, (m >= thr).astype(np.int8), average=None, zero_division=0)
    print(f"\nBEST w=({ws},{wb},{wl})  OOF {score:.4f} (shipped 13.68)  "
          f"art {art:.4f}  qual {qual:.4f}")
    print("  per-label:", {k: round(float(v), 3) for k, v in zip(DEFECTS, per_label)})
    print("  artefact thr:", " ".join(f"{t:.2f}" for t in thr))
    print("  quality  thr:", " ".join(f"{t:.2f}" for t in qt))

    # ---- half-split transfer check (tune on A, score on B) ----
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    half_a, half_b = perm[: n // 2], perm[n // 2:]
    m_a, m_b = m[half_a], m[half_b]
    thr_a, _ = tune_artefact(m_a, y[half_a], anneal=2000)
    qt_a, _ = tune_quality(m_a, yq[half_a])
    sb, _, _ = dual_score(m_b, y[half_b], yq[half_b], thr_a, qt_a)
    # reference: shipped-style coupled tuning on A applied to B
    thr_c, _ = metric.tune_thresholds(m_a, y[half_a], yq[half_a])
    dc = (m_b >= thr_c).astype(np.int8)
    sc = 10 * artefact_f1w(y[half_b], dc) + 10 * quality_f1(yq[half_b], metric.derive_quality(dc))
    print(f"half-split transfer: dual {sb:.4f} vs coupled {sc:.4f}  (delta {sb - sc:+.3f})")

    blob = {
        "weights": {"s": float(ws), "b": float(wb), "l4": float(wl)},
        "thresholds": {k: float(t) for k, t in zip(DEFECTS, thr)},
        "quality_thresholds": {k: float(t) for k, t in zip(DEFECTS, qt)},
        "oof_score": float(score), "oof_artefact": float(art), "oof_quality": float(qual),
        "half_split_dual": float(sb), "half_split_coupled": float(sc),
        "per_label_f1": {k: float(v) for k, v in zip(DEFECTS, per_label)},
        "seed": 0,
    }
    np.save(out / "oof_fuse3.npy", m)

    if args.dry_run:
        (out / "fuse3_dry.json").write_text(json.dumps(blob, indent=2))
        print(f"[dry-run] wrote {out / 'fuse3_dry.json'}")
        return 0

    # ---- test probabilities ----
    te_ids = np.load(cache / "dinos_test" / "item_ids.npy")
    te = {
        "s": load_oof(cache, "probs_test_s.npy"),
        "b": load_oof(cache, "probs_test_b.npy"),
        "l4": load_oof(args.l4_dir, "probs_test_l4.npy"),
    }
    l4_te_ids = np.load(args.l4_dir / "item_ids_test.npy")
    if list(l4_te_ids) != list(te_ids):
        raise SystemExit("L4 test row order differs from the cache order; realign before fusing")
    probs_te = ws * te["s"] + wb * te["b"] + wl * te["l4"]
    d_te = (probs_te >= thr).astype(np.int8)
    q_te = ((probs_te >= qt).any(axis=1) == 0).astype(np.int8)
    metric.write_submission(args.submission, list(te_ids), d_te, q_te)
    blob["pred_positives"] = {k: int(v) for k, v in zip(DEFECTS, d_te.sum(axis=0))}
    blob["pred_quality_good"] = int(q_te.sum())
    blob["submission_md5"] = hashlib.md5(args.submission.read_bytes()).hexdigest()
    (out / "fuse3.json").write_text(json.dumps(blob, indent=2))
    np.save(out / "probs_test_fuse3.npy", probs_te)
    print(f"\nwrote {args.submission} (md5 {blob['submission_md5']})")
    print("  positives:", blob["pred_positives"], " quality-good:", blob["pred_quality_good"])
    print(f"  config -> {out / 'fuse3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
