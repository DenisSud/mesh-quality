"""Build a self-contained HTML data explorer for the mesh-quality task.

Usage (from the repo root):

    devenv shell -- uv run python main/mesh-quality/tools/explorer/build_html.py [out.html]

Needs ``main/mesh-quality/cache/geometry_{train,test}.csv`` (see
``mesh_quality.geometry``); if they are missing it falls back to extracting
features for the sampled items only. Thumbnails and decimated meshes are cached
under ``tools/explorer/payload_cache/`` so rebuilds take seconds.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/home/denis/dev/aiijc")
TASK = ROOT / "main/mesh-quality"
DATA = TASK / "data"
CACHE = TASK / "cache"
TMP = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mesh_quality_explorer.html")

sys.path.insert(0, str(TASK / "src"))
from mesh_quality.geometry import FEATURE_NAMES, extract_features  # noqa: E402

LABELS = ["abstract", "artifacts", "intersection", "lowpoly", "noisy", "open", "partial", "scale", "set", "simple"]


# --------------------------------------------------------------- payloads ---
PAYLOAD_CACHE = CACHE / "explorer_payload"
PAYLOAD_CACHE.mkdir(parents=True, exist_ok=True)


def cached_thumb(item_id: str, split: str, width: int, quality: int) -> str:
    path = PAYLOAD_CACHE / f"thumb_{split}_{item_id}_{width}_{quality}.txt"
    if not path.exists():
        path.write_text(thumb(item_id, split, width, quality))
    return path.read_text()


def cached_mesh(item_id: str, split: str, max_faces: int = 4000) -> dict:
    path = PAYLOAD_CACHE / f"mesh_{split}_{item_id}_{max_faces}.json"
    if not path.exists():
        path.write_text(json.dumps(mesh_payload(item_id, split, max_faces)))
    return json.loads(path.read_text())


# --------------------------------------------------------------- images -----
def thumb(item_id: str, split: str, width: int, quality: int) -> str:
    path = DATA / split / f"{item_id}.png"
    with Image.open(path) as im:
        im = im.convert("RGB")
        h = round(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- meshes ----
def mesh_payload(item_id: str, split: str, max_faces: int = 4000) -> dict:
    """Decimated preview mesh: strided subsample, then grid vertex clustering."""
    with np.load(DATA / split / f"{item_id}.npz") as d:
        v = np.asarray(d["vertices"], dtype=np.float64)
        f = np.asarray(d["faces"], dtype=np.int64)
    f = f[:: max(1, len(f) // 200_000)]
    for grid in (96, 64, 48, 32, 24, 16):
        pv, pf = _cluster(v, f, grid)
        if len(pf) <= max_faces:
            break
    lo, hi = pv.min(axis=0), pv.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    q = np.round((pv - lo) / span * 65535).astype(np.uint16)
    dtype = "u16" if len(pv) <= 65535 else "u32"
    idx = pf.astype(np.uint16 if dtype == "u16" else np.uint32)
    return {
        "nv": int(len(pv)),
        "nf": int(len(pf)),
        "dtype": dtype,
        "pos": base64.b64encode(q.tobytes()).decode(),
        "idx": base64.b64encode(idx.tobytes()).decode(),
    }


def _cluster(v: np.ndarray, f: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that fall into the same grid cell; drop degenerate/duplicate faces."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    g = np.clip(np.round((v - lo) / span * (grid - 1)), 0, grid - 1).astype(np.int64)
    key = g[:, 0] + g[:, 1] * grid + g[:, 2] * grid * grid
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    nv2 = int(inv.max()) + 1
    count = np.bincount(inv, minlength=nv2).astype(np.float64)
    pos = np.stack([np.bincount(inv, weights=v[:, k], minlength=nv2) / count for k in range(3)], axis=1)
    faces = inv[f]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    if len(faces) == 0:
        return pos, faces
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return pos, faces[np.sort(first)]


# --------------------------------------------------------------- sampling ---
def pick_gallery(frame: pd.DataFrame) -> list[str]:
    """~30 items: a few clean objects plus a few per defect, spread over mesh sizes."""
    chosen: list[str] = []
    ids = frame["item_id"].values
    ndef = frame[LABELS].sum(axis=1).values
    nf = frame["n_faces"].values
    for label in LABELS:
        idx = np.flatnonzero(frame[label].values == 1)
        if not len(idx):
            continue
        idx = idx[np.lexsort((nf[idx], ndef[idx]))]  # fewest co-defects first, then smallest mesh
        chosen += [str(ids[i]) for i in (idx[0], idx[len(idx) // 2], idx[-2], idx[-1])]
    clean = frame["item_id"].values[(frame["quality"] == 1).values]
    rng = np.random.default_rng(0)
    chosen += [str(i) for i in rng.choice(clean, size=8, replace=False)]
    return list(dict.fromkeys(chosen))


def pick_scatter(frame: pd.DataFrame, n_rand: int = 150) -> list[str]:
    """Random spread plus a handful of positives for every defect (rare labels included)."""
    rng = np.random.default_rng(7)
    ids = frame["item_id"].values
    picks = [str(i) for i in rng.choice(ids, size=n_rand, replace=False)]
    for label in LABELS:
        idx = np.flatnonzero(frame[label].values == 1)
        if len(idx):
            picks += [str(ids[i]) for i in rng.choice(idx, size=min(12, len(idx)), replace=False)]
    return list(dict.fromkeys(picks))


# -------------------------------------------------------------- analyses ----
def _design(geo: pd.DataFrame, train: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Feature matrix + label frame restricted to items present in the geometry cache."""
    frame = geo.set_index("item_id")
    ids = [i for i in train["item_id"] if i in frame.index]
    X = frame.loc[ids, list(FEATURE_NAMES)].to_numpy(float)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
    return X, train.set_index("item_id").loc[ids].reset_index()


def auc_table(geo: pd.DataFrame, train: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    X, labels_frame = _design(geo, train)
    labels = LABELS + ["quality"]
    values = []
    for i, feat in enumerate(FEATURE_NAMES):
        col = X[:, i]
        row = []
        for name in labels:
            y = labels_frame[name].to_numpy()
            row.append(0.5 if y.min() == y.max() else float(roc_auc_score(y, col)))
        values.append(row)
    return {"labels": labels, "features": list(FEATURE_NAMES), "values": values, "n": len(labels_frame)}


def geometry_baseline(geo: pd.DataFrame, train: pd.DataFrame, folds: int = 5) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    X, labels_frame = _design(geo, train)
    out = []
    targets = LABELS + ["quality"]
    print(f"  baseline on {len(labels_frame)} items", flush=True)
    for name in targets:
        y = labels_frame[name].to_numpy()
        prob = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        for tr, va in skf.split(X, y):
            model = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, class_weight="balanced", random_state=0,
            ).fit(X[tr], y[tr])
            prob[va] = model.predict_proba(X[va])[:, 1]
        pred = prob > 0.5
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out.append({"label": name, "pos": float(y.mean() * 100), "prec": prec, "rec": rec, "f1": f1})
    return out


# ------------------------------------------------------------------ stats ---
def basic_stats(train: pd.DataFrame, geo: pd.DataFrame) -> dict:
    labels = LABELS + ["quality"]
    frame = train[labels].to_numpy()
    cooc = (frame.T @ frame).tolist()
    combos = Counter()
    for row in train[LABELS].to_numpy():
        combos[",".join(LABELS[i] for i in np.flatnonzero(row))] += 1
    combos.pop("", None)
    top = combos.most_common(12)
    merged = geo.set_index("item_id").loc[train["item_id"]]
    notes = [
        f"<b>quality == 1 ⟺ all ten defect columns are 0</b> — {int((train['quality'] == 1).sum())} clean objects, no exceptions.",
        f"<b>{train['quality'].mean() * 100:.1f}%</b> of objects are clean; the metric's quality half is a rare-class F1.",
        "Defects are mostly single-label: "
        + ", ".join(f"{d.split(',')[0] if d else 'clean'} {c}" for d, c in top[:5])
        + " dominate.",
        f"Mesh size spans <b>{int(merged['n_faces'].min()):,}</b>–<b>{int(merged['n_faces'].max()):,}</b> triangles "
        f"(median {int(merged['n_faces'].median()):,}) — geometry features must be size-normalised.",
        f"<b>{(merged['n_components'] > 1).mean() * 100:.1f}%</b> of meshes have more than one connected component; "
        f"<b>{(merged['boundary_edge_frac'] > 1e-6).mean() * 100:.1f}%</b> have boundary edges (open surfaces).",
    ]
    return {
        "prevalence": {l: float(train[l].mean() * 100) for l in LABELS},
        "quality_pct": float(train["quality"].mean() * 100),
        "cooc_with_quality": cooc,
        "combos": top,
        "combos_max": top[0][1] if top else 1,
        "notes": notes,
    }


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    geo_path = CACHE / "geometry_train.csv"
    full_cache = geo_path.exists() and (CACHE / "geometry_test.csv").exists()
    if full_cache:
        geo = pd.concat([pd.read_csv(geo_path), pd.read_csv(CACHE / "geometry_test.csv")], ignore_index=True)
        merged = train.merge(geo, on="item_id", how="left")
        gallery_ids = pick_gallery(merged)
        scatter_ids = pick_scatter(merged)
    else:  # partial: pick without features, then extract features for the sample only
        print("geometry cache incomplete — sampling on labels alone")
        sizes = {i: (DATA / "train" / f"{i}.npz").stat().st_size for i in train["item_id"]}
        coarse = train.assign(n_faces=train["item_id"].map(sizes))
        gallery_ids = pick_gallery(coarse)
        scatter_ids = pick_scatter(coarse)
        ids = list(dict.fromkeys(scatter_ids + gallery_ids))
        geo = extract_features([(i, str(DATA / "train" / f"{i}.npz")) for i in ids], workers=3)
        merged = train.merge(geo, on="item_id", how="left")
    ids = list(dict.fromkeys(scatter_ids + [i for i in gallery_ids if i not in scatter_ids]))
    label_by_id = train.set_index("item_id")

    geo_idx = geo.set_index("item_id")
    items = []
    for iid in ids:
        split = "train" if iid in label_by_id.index else "test"
        row = label_by_id.loc[iid] if split == "train" else None
        feats = {f: (float(v) if np.isfinite(v) else None) for f, v in geo_idx.loc[iid].items() if f != "split"}
        item = {
            "id": iid,
            "split": split,
            "labels": [int(row[l]) for l in LABELS] if row is not None else [0] * 10,
            "quality": int(row["quality"]) if row is not None else 0,
            "feats": feats,
            "thumb": iid,
            "mesh": iid if iid in gallery_ids else None,
            "open_path": f"file://{DATA}/{split}/{iid}.png",
        }
        items.append(item)

    print(f"embedding {len(items)} items ({len(gallery_ids)} with mesh)")
    thumbs = {}
    for n, it in enumerate(items):
        big = it["mesh"] is not None
        thumbs[it["id"]] = cached_thumb(it["id"], it["split"], 1024 if big else 512, 76 if big else 66)
        if n % 60 == 0:
            print(f"  thumb {n}/{len(items)}", flush=True)
    meshes = {iid: cached_mesh(iid, "train") for iid in gallery_ids}
    print("meshes done")

    stats = basic_stats(train, geo)
    auc = auc_table(geo, train)
    baseline = geometry_baseline(geo, train)
    print("analyses done")

    payload = {
        "meta": {
            "n_train": len(train), "n_test": len(test),
            "labels": LABELS, "features": list(FEATURE_NAMES),
            "sample": len(items), "full_cache": bool(full_cache),
        },
        "items": items,
        "thumbs": thumbs,
        "meshes": meshes,
        "stats": stats,
        "auc": auc,
        "baseline": baseline,
    }

    html = (TMP / "shell.html").read_text()
    html = html.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False))
    html = html.replace("__APP__", (TMP / "app.js").read_text())
    html = html.replace('<div class="pills" id="pills"></div>', pills(payload["meta"]))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


def pills(meta: dict) -> str:
    cache = "full train+test" if meta.get("full_cache") else f'sample of {meta.get("sample")}'
    return (
        '<div class="pills" id="pills">'
        '<span class="pill">train <b>8 964</b> objects</span>'
        '<span class="pill">test <b>669</b> objects</span>'
        '<span class="pill">per object <b>6 renders</b> (1536×1024) + <b>mesh</b> (.npz)</span>'
        '<span class="pill">metric <b>10·F1(quality) + 10·F1_weighted(defects)</b></span>'
        f'<span class="pill">geometry features from <b>{cache}</b></span>'
        '</div>'
    )


if __name__ == "__main__":
    main()
