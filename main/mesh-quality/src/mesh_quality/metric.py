"""Task metric and per-label threshold tuning.

``f1_final = 10 * F1(quality, pos_label=1) + 10 * f1_score(defects, average='weighted')``

``quality`` is exactly "no defect" in the data (verified on all 8964 train rows),
so predictions derive it from the thresholded defect matrix instead of predicting
it independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DEFECTS: tuple[str, ...] = (
    "abstract",
    "artifacts",
    "intersection",
    "lowpoly",
    "noisy",
    "open",
    "partial",
    "scale",
    "set",
    "simple",
)
COLUMNS: tuple[str, ...] = ("item_id", *DEFECTS, "quality")
DEFAULT_THRESHOLDS = np.full(len(DEFECTS), 0.5)


def derive_quality(defects: np.ndarray) -> np.ndarray:
    """``quality = 1`` iff no defect is predicted (the data identity)."""
    return (np.asarray(defects).sum(axis=1) == 0).astype(np.int8)


def score_predictions(
    y_defects: np.ndarray,
    y_quality: np.ndarray,
    d_pred: np.ndarray,
    q_pred: np.ndarray,
) -> dict:
    """Compute the competition score plus per-label diagnostics."""
    y_defects = np.asarray(y_defects)
    d_pred = np.asarray(d_pred)
    per_label = f1_score(y_defects, d_pred, average=None, zero_division=0)
    support = y_defects.sum(axis=0)
    artefact = float(f1_score(y_defects, d_pred, average="weighted", zero_division=0))
    quality = float(f1_score(y_quality, q_pred, pos_label=1, zero_division=0))
    return {
        "score": 10.0 * artefact + 10.0 * quality,
        "artefact_f1_weighted": artefact,
        "quality_f1": quality,
        "per_label": {name: float(v) for name, v in zip(DEFECTS, per_label)},
        "support": {name: int(v) for name, v in zip(DEFECTS, support)},
        "pred_positives": {name: int(v) for name, v in zip(DEFECTS, d_pred.sum(axis=0))},
    }


def tune_thresholds(
    probs: np.ndarray,
    y_defects: np.ndarray,
    y_quality: np.ndarray,
    grid: np.ndarray | None = None,
    passes: int = 3,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Coordinate ascent on the per-label decision thresholds for the task metric.

    Rare labels trade off only weakly against each other, so greedy per-label
    search with a couple of refinement passes is enough.
    """
    probs = np.asarray(probs)
    if grid is None:
        grid = np.concatenate([[0.01, 0.02, 0.03, 0.05, 0.075], np.arange(0.1, 0.96, 0.05)])
    thr = np.full(probs.shape[1], 0.5)
    best = score_predictions(
        y_defects, y_quality, (probs >= thr).astype(np.int8), derive_quality((probs >= thr).astype(np.int8))
    )["score"]
    for _ in range(passes):
        improved = False
        for k in range(probs.shape[1]):
            keep, keep_score = thr[k], best
            for v in grid:
                cand = thr.copy()
                cand[k] = v
                d = (probs >= cand).astype(np.int8)
                s = score_predictions(y_defects, y_quality, d, derive_quality(d))["score"]
                if s > keep_score + 1e-9:
                    keep, keep_score = v, s
            if keep != thr[k]:
                thr[k] = keep
                best = keep_score
                improved = True
                if verbose:
                    print(f"    thr[{DEFECTS[k]}] -> {keep:.3f}  score {best:.4f}")
        if not improved:
            break
    d = (probs >= thr).astype(np.int8)
    return thr, score_predictions(y_defects, y_quality, d, derive_quality(d))


def read_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: submission missing columns {missing}")
    return df[list(COLUMNS)]


def write_submission(
    path: str | Path,
    item_ids: list[str] | np.ndarray,
    d_pred: np.ndarray,
    q_pred: np.ndarray | None = None,
) -> Path:
    d_pred = np.asarray(d_pred).astype(np.int8)
    if q_pred is None:
        q_pred = derive_quality(d_pred)
    out = pd.DataFrame({"item_id": list(item_ids)})
    for k, name in enumerate(DEFECTS):
        out[name] = d_pred[:, k]
    out["quality"] = np.asarray(q_pred).astype(np.int8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
