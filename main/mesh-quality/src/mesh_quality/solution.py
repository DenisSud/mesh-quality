"""Solution entry points for the 3D Mesh Quality Control task.

Pipeline:

1. ``features``  -- cache frozen DINOv3 render features (``cache/dino*_*`` dirs)
2. ``train``     -- cross-validated attentive probe on cached features, threshold
   tuning for the task metric, then full-data refit
3. ``predict``   -- write ``submission.csv``
4. ``score``     -- evaluate a labelled submission with the task metric
5. ``visualize`` -- per-label patch attribution for one item
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import images, metric, model


def features(args: argparse.Namespace) -> None:
    """Cache frozen DINOv3 render features for the requested splits."""
    cache_dir = args.task_dir / "cache"
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        print(f"[features] {split}: {images.MODELS[args.model]} -> {images.POOLED_GRID}x{images.POOLED_GRID} pooled grid", flush=True)
        images.extract_split(
            data_dir=args.data_dir,
            split=split,
            cache_dir=cache_dir,
            model_key=args.model,
            limit=args.limit,
        )


def _probe_config(ds: model.Dataset, args: argparse.Namespace) -> model.ProbeConfig:
    return model.ProbeConfig(
        dim=ds.dim,
        n_geom=int(ds.geom.shape[1]),
        grid=ds.grid,
        d=args.d,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )


def _report(tag: str, res: dict) -> None:
    print(f"[{tag}] score {res['score']:.3f}  artefact F1w {res['artefact_f1_weighted']:.4f}  quality F1 {res['quality_f1']:.4f}")
    for name in metric.DEFECTS:
        print(
            f"    {name:<13} F1 {res['per_label'][name]:.3f}  support {res['support'][name]:<5} "
            f"pred {res['pred_positives'][name]}"
        )


def _geometry_probe_path(cache_dir) -> "Path":
    return cache_dir / "probe_geometry.pkl"


def train_geometry(args: argparse.Namespace) -> None:
    """Fallback solution: per-label HistGB on geometry features only (~10.3/20 at seed 0)."""
    import joblib

    cache_dir = args.task_dir / "cache"
    ds = model.load_geometry(cache_dir, args.data_dir, "train", limit=args.limit)
    median, scale = model.geometry_stats(ds.geom)
    X = ds.standardise(median, scale)
    y = ds.labels
    assert y is not None
    print(f"[train] geometry-only HistGB: {len(ds)} items, {ds.geom.shape[1]} features")
    oof = model.histgb_oof(X, y, k=args.folds, seed=args.seed)
    thr, res = metric.tune_thresholds(oof, y, metric.derive_quality(y), verbose=True)
    _report("cv geometry-only HistGB", res)
    print("[train] thresholds " + json.dumps({n: round(float(t), 4) for n, t in zip(metric.DEFECTS, thr)}))
    np.save(cache_dir / "oof_geometry.npy", oof)
    models, _ = model.histgb_fit(X, y, seed=args.seed)
    path = _geometry_probe_path(cache_dir)
    joblib.dump(
        {
            "models": models,
            "thresholds": thr,
            "median": median,
            "scale": scale,
            "geom_names": ds.geom_names,
            "cv": res,
            "version": 1,
        },
        path,
    )
    print(f"[train] wrote {path}")


def train(args: argparse.Namespace) -> None:
    """Fit the probe: CV for honest scores/thresholds, then a full-data refit."""
    if args.model == "geometry":
        return train_geometry(args)

    import torch

    cache_dir = args.task_dir / "cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] model={args.model} device={device}")

    ds = model.load_dataset(cache_dir, args.data_dir, args.model, "train", limit=args.limit)
    print(f"[train] {len(ds)} items, image dim {ds.dim}, {ds.geom.shape[1]} geometry features")
    median, scale = model.geometry_stats(ds.geom)
    geom_z = ds.standardise(median, scale)
    cfg = _probe_config(ds, args)
    y = ds.labels
    assert y is not None

    oof, _ = model.run_cv(
        ds,
        cfg,
        geom_z,
        k=args.folds,
        fold_seed=args.seed,
        model_seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        log_every=10,
    )
    thr, res = metric.tune_thresholds(oof, y, metric.derive_quality(y), verbose=True)
    _report("cv image+geometry probe", res)
    print("[train] thresholds " + json.dumps({n: round(float(t), 4) for n, t in zip(metric.DEFECTS, thr)}))
    if args.limit is not None:
        print("[train] --limit run: smoke test only, no artifacts written")
        return
    np.save(cache_dir / f"oof_probe_{args.model}.npy", oof)

    # Full-data refit (optionally several seeds for a small ensemble).
    states = []
    for s in range(args.full_seeds):
        print(f"[train] full-data refit seed {args.seed + s}")
        full, hist = model.train_model(
            ds,
            cfg,
            geom_z,
            np.arange(len(ds)),
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            wd=args.wd,
            seed=args.seed + s,
            device=device,
        )
        states.append(full.state_dict())

    ckpt = {
        "cfg": cfg.__dict__,
        "model_key": args.model,
        "states": states,
        "thresholds": thr.tolist(),
        "geom_median": median,
        "geom_scale": scale,
        "cv": res,
        "version": 1,
    }
    path = cache_dir / f"probe_{args.model}.pt"
    torch.save(ckpt, path)
    print(f"[train] wrote {path}")


def _load_probe(cache_dir, model_key: str, device: str | None):
    import torch

    ckpt = torch.load(cache_dir / f"probe_{model_key}.pt", map_location=device, weights_only=False)
    cfg = model.ProbeConfig(**ckpt["cfg"])
    state = model.AttentiveProbe(cfg)
    state.load_state_dict(ckpt["states"][0])
    return ckpt, cfg, state


def fusion_path(cache_dir, tag: str):
    return Path(cache_dir) / f"fuse_{tag}.json"


def load_fusion(cache_dir, tag: str) -> dict:
    """Load the fusion metadata written by :func:`fuse` (models, weights, thresholds)."""
    path = fusion_path(cache_dir, tag)
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run `fuse --models {tag}` first (it records the fused thresholds)")
    return json.loads(path.read_text())


def predict_probe(cache_dir, data_dir, model_key: str, device: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Test probabilities of one probe, averaged over its refit states."""
    ckpt, cfg, _ = _load_probe(cache_dir, model_key, device)
    ds = model.load_dataset(cache_dir, data_dir, ckpt["model_key"], "test")
    if cfg.grid != ds.grid:
        raise SystemExit(f"probe_{model_key}.pt expects a {cfg.grid}x{cfg.grid} token grid, cache has {ds.grid}x{ds.grid}")
    geom_z = ds.standardise(np.asarray(ckpt["geom_median"]), np.asarray(ckpt["geom_scale"]))
    probs = np.zeros((len(ds), len(metric.DEFECTS)), dtype=np.float32)
    for state in ckpt["states"]:
        net = model.AttentiveProbe(cfg).to(device)
        net.load_state_dict(state)
        probs += model.predict_probs(net, ds, geom_z, np.arange(len(ds)), device=device)
    probs /= len(ckpt["states"])
    return probs, np.asarray(ds.item_ids)


def predict(args: argparse.Namespace) -> None:
    """Average the probe ensemble over the test split and write the submission.

    ``--model s`` runs one probe; ``--models s,b`` averages several and applies
    the fused thresholds recorded by :func:`fuse` (the pair is what the shipped
    candidate was built from, so the notebook must reproduce both the same way).
    """
    keys = [k.strip() for k in (getattr(args, "models", None) or args.model).split(",") if k.strip()]
    if keys == ["geometry"]:
        return predict_geometry(args)
    if "geometry" in keys:
        raise SystemExit("--models: fusion is only supported between image probes, not 'geometry'")

    import torch

    cache_dir = args.task_dir / "cache"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    weights = None
    if len(keys) > 1:
        fusion = load_fusion(cache_dir, "+".join(keys))
        if fusion["models"] != keys:
            raise SystemExit(f"fusion_{'+'.join(keys)}.json covers {fusion['models']}, not {keys}")
        weights = np.asarray(fusion["weights"], dtype=np.float32)
    if weights is None:
        weights = np.full(len(keys), 1.0 / len(keys), dtype=np.float32)

    probs, item_ids = None, None
    for key, weight in zip(keys, weights):
        p, ids = predict_probe(cache_dir, args.data_dir, key, device)
        probs = weight * p if probs is None else probs + weight * p
        item_ids = ids
    assert probs is not None and item_ids is not None

    if len(keys) == 1:
        ckpt = torch.load(cache_dir / f"probe_{keys[0]}.pt", map_location="cpu", weights_only=False)
        thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)
        probs_path = cache_dir / f"probs_test_{keys[0]}.npy"
    else:
        tag = "+".join(keys)
        thresholds = np.asarray(load_fusion(cache_dir, tag)["thresholds"], dtype=np.float32)
        probs_path = cache_dir / f"probs_test_fuse_{tag}.npy"

    d_pred = (probs >= thresholds).astype(np.int8)
    path = metric.write_submission(args.submission, item_ids, d_pred)
    np.save(probs_path, probs)
    label = "+".join(keys)
    print(f"[predict] {path}: {len(item_ids)} rows, model {label}, positives per label " + str(d_pred.sum(0).tolist()))
    log_submission(
        args.task_dir,
        model=label,
        probs_path=probs_path,
        thresholds=thresholds,
        positives=d_pred.sum(0),
        submission_path=path,
    )


def predict_geometry(args: argparse.Namespace) -> None:
    """Predict with the geometry-only fallback and write the submission."""
    import joblib

    cache_dir = args.task_dir / "cache"
    payload = joblib.load(_geometry_probe_path(cache_dir))
    ds = model.load_geometry(cache_dir, args.data_dir, "test")
    X = ds.standardise(np.asarray(payload["median"]), np.asarray(payload["scale"]))
    probs = np.stack([m.predict_proba(X)[:, 1] for m in payload["models"]], axis=1).astype(np.float32)
    thr = np.asarray(payload["thresholds"], dtype=np.float32)
    d_pred = (probs >= thr).astype(np.int8)
    path = metric.write_submission(args.submission, ds.item_ids, d_pred)
    np.save(cache_dir / "probs_test_geometry.npy", probs)
    print(f"[predict] {path}: {len(ds)} rows, positives per label " + str(d_pred.sum(0).tolist()))


def _parse_thresholds(text: str | None) -> np.ndarray | None:
    """Parse ``--thresholds`` as 10 comma-separated values in label order."""
    if text is None:
        return None
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != len(metric.DEFECTS):
        raise SystemExit(f"--thresholds needs {len(metric.DEFECTS)} comma-separated values in label order")
    return np.asarray([float(p) for p in parts], dtype=np.float32)


# --------------------------------------------------------------------------- #
# submission bookkeeping
# --------------------------------------------------------------------------- #

#: one row per uploaded candidate: what produced it, from which code state.
LOG_COLUMNS = [
    "id",
    "ts",
    "lb_score",
    "place",
    "commit",
    "model",
    "probs",
    "probs_md5",
    "thresholds",
    "positives",
    "submission",
    "submission_md5",
    "notes",
]


def _read_log(log: Path) -> list[dict]:
    """Read ``submissions.csv`` normalised to ``LOG_COLUMNS``.

    Tolerates a trailing comma in the header (which makes ``csv.DictReader``
    yield an extra ``''`` field) and rows written by older revisions that had
    fewer columns. Both cases used to abort the write-back with a ValueError.
    """
    import csv

    if not log.exists():
        return []
    with log.open() as f:
        raw = list(csv.DictReader(f))
    return [{key: (row.get(key) or "") for key in LOG_COLUMNS} for row in raw]


def _md5(path) -> str:
    import hashlib

    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _git_rev(task_dir) -> str:
    """Short HEAD hash of the repo, marked ``-dirty`` when the task dir has edits."""
    import subprocess

    root = Path(task_dir).resolve().parents[1]
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", str(Path(task_dir).resolve())],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - git is optional bookkeeping
        return "unknown"


def _write_log(log: Path, rows: list[dict]) -> None:
    """Rewrite ``submissions.csv`` atomically (tmp file + replace).

    The log used to be truncated in place before being serially rewritten, so a
    failed write destroyed every earlier row; keep the write all-or-nothing.
    """
    import csv
    import os
    import tempfile

    log.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=log.parent, prefix=".submissions_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, log)
    except BaseException:
        os.unlink(tmp)
        raise


def log_submission(
    task_dir,
    *,
    model: str,
    probs_path,
    thresholds,
    positives,
    submission_path,
    notes: str = "",
) -> str:
    """Append one row to ``submissions.csv``; returns the assigned id."""
    import datetime as dt

    log = Path(task_dir) / "submissions.csv"
    rows = _read_log(log)
    sid = f"s{len(rows) + 1:02d}"
    row = {
        "id": sid,
        "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lb_score": "",
        "place": "",
        "commit": _git_rev(task_dir),
        "model": model,
        "probs": str(probs_path),
        "probs_md5": _md5(probs_path),
        "thresholds": " ".join(f"{float(t):.3f}" for t in thresholds),
        "positives": " ".join(str(int(p)) for p in positives),
        "submission": str(Path(submission_path).name),
        "submission_md5": _md5(submission_path),
        "notes": notes,
    }
    rows.append(row)
    _write_log(log, rows)
    print(f"[log] {log}: {sid} ({row['commit']})")
    return sid


def logscore(args: argparse.Namespace) -> None:
    """Record the leaderboard score/place of a logged submission."""

    log = args.task_dir / "submissions.csv"
    rows = _read_log(log)
    if not rows:
        raise SystemExit(f"{log} does not exist yet")
    hits = [r for r in rows if r["id"] == args.id]
    if not hits:
        raise SystemExit(f"{args.id}: no such entry in {log}")
    row = hits[-1]
    row["lb_score"] = f"{args.score:.4f}" if args.score is not None else row["lb_score"]
    row["place"] = str(args.place) if args.place is not None else row["place"]
    if args.notes:
        row["notes"] = (row["notes"] + " | " + args.notes).strip(" |")
    _write_log(log, rows)
    print(f"[log] {args.id}: score={row['lb_score'] or '-'} place={row['place'] or '-'}")


def resubmit(args: argparse.Namespace) -> None:
    """Re-threshold a saved test-probability matrix into a submission (no model)."""
    import pandas as pd

    probs = np.load(args.probs).astype(np.float32)
    ids = pd.read_csv(args.data_dir / "test.csv")["item_id"].astype(str).to_numpy()
    if len(probs) != len(ids):
        raise SystemExit(f"{args.probs}: {len(probs)} rows do not match {len(ids)} test ids")
    thr = _parse_thresholds(args.thresholds)
    if thr is None:
        raise SystemExit("--thresholds is required (10 values, label order)")
    d_pred = (probs >= thr).astype(np.int8)
    path = metric.write_submission(args.submission, ids, d_pred)
    print(f"[resubmit] {path}: {len(ids)} rows, positives per label " + str(d_pred.sum(0).tolist()))
    log_submission(
        args.task_dir,
        model=args.model,
        probs_path=args.probs,
        thresholds=thr,
        positives=d_pred.sum(0),
        submission_path=path,
        notes=args.notes or "",
    )


def fuse(args: argparse.Namespace) -> None:
    """Average test probabilities of several trained probes and report fused thresholds.

    The fused thresholds are tuned on the average of the per-model OOF matrices,
    which is the honest counterpart of the test-side average.
    """
    import pandas as pd

    cache_dir = args.task_dir / "cache"
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if len(keys) < 2:
        raise SystemExit("--models needs at least two comma-separated keys, e.g. s,b")
    weights = [float(w) for w in (args.weights or "").replace(";", ",").split(",") if w.strip()]
    if not weights:
        weights = [1.0] * len(keys)
    if len(weights) != len(keys):
        raise SystemExit(f"--weights needs {len(keys)} values for {keys}")
    w = np.asarray(weights, dtype=np.float32)
    w /= w.sum()

    probs = np.zeros_like(np.load(cache_dir / f"probs_test_{keys[0]}.npy"), dtype=np.float32)
    oof = np.zeros_like(np.load(cache_dir / f"oof_probe_{keys[0]}.npy"), dtype=np.float32)
    for key, wi in zip(keys, w):
        p = np.load(cache_dir / f"probs_test_{key}.npy").astype(np.float32)
        o = np.load(cache_dir / f"oof_probe_{key}.npy").astype(np.float32)
        if p.shape != probs.shape or o.shape != oof.shape:
            raise SystemExit(f"{key}: shapes {p.shape}/{o.shape} do not match {probs.shape}/{oof.shape}")
        probs += wi * p
        oof += wi * o
    print("[fuse] weights " + " ".join(f"{k}={wi:.2f}" for k, wi in zip(keys, w)))

    labels = pd.read_csv(args.data_dir / "train.csv").set_index("item_id")
    train_ids = np.asarray(images.load_cache(cache_dir, keys[0], "train").item_ids)
    for key in keys[1:]:
        other = np.asarray(images.load_cache(cache_dir, key, "train").item_ids)
        if not np.array_equal(train_ids, other):
            raise SystemExit(f"{key}: train item order differs from {keys[0]}")
    y = labels.loc[list(train_ids), list(metric.DEFECTS)].to_numpy(dtype=np.int8)
    thr, res = metric.tune_thresholds(oof, y, metric.derive_quality(y), verbose=True)
    tag = "+".join(keys)
    _report(f"fused oof {tag}", res)
    np.save(cache_dir / f"probs_test_fuse_{tag}.npy", probs)
    np.save(cache_dir / f"oof_fuse_{tag}.npy", oof)
    payload = {
        "models": keys,
        "weights": [float(x) for x in w],
        "thresholds": [float(t) for t in thr],
        "cv": res,
        "probs": f"cache/probs_test_fuse_{tag}.npy",
        "oof": f"cache/oof_fuse_{tag}.npy",
    }
    meta = fusion_path(cache_dir, tag)
    meta.write_text(json.dumps(payload, indent=2))
    print(f"[fuse] wrote {cache_dir / f'probs_test_fuse_{tag}.npy'}")
    print(f"[fuse] wrote {meta} (thresholds travel with the fused artifact)")
    print("[fuse] thresholds " + ",".join(f"{float(t):.3f}" for t in thr))
    print(f"[fuse] reproduce with: predict --models {','.join(keys)}")


def score(args: argparse.Namespace) -> None:
    """Score a submission that covers items present in the local ground truth."""
    import pandas as pd

    truth = pd.read_csv(args.data_dir / "train.csv").set_index("item_id")
    sub = metric.read_submission(args.submission)
    ids = [i for i in sub["item_id"] if i in truth.index]
    if not ids:
        raise SystemExit(f"{args.submission}: no rows match labelled items in {args.data_dir/'train.csv'}")
    sub = sub.set_index("item_id").loc[ids]
    truth = truth.loc[ids]
    y_defects = truth[list(metric.DEFECTS)].to_numpy(dtype=np.int8)
    d_pred = sub[list(metric.DEFECTS)].to_numpy(dtype=np.int8)
    res = metric.score_predictions(y_defects, metric.derive_quality(y_defects), d_pred, sub["quality"].to_numpy())
    _report(f"score ({len(ids)} labelled rows)", res)


def visualize(args: argparse.Namespace) -> None:
    """Per-label attribution over the DINOv3 patch grid of one item."""
    from . import visualize as viz

    viz.run(args)


def deliver(args: argparse.Namespace) -> None:
    """Stage the Colab reproduction bundle (code + probe + references) as a zip."""
    from . import deliver as bundle

    bundle.stage(args)
