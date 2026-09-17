"""Frozen-feature attentive probe for the mesh-defect labels.

Image side: cached DINOv3 summaries (per-tile mean/max + pooled patch grid).
Mesh side: the hand-computed geometry features from ``geometry.py`` plus a few
scale-invariant derivations, fed as one ``[GEOM]`` token.

The probe follows the DINOv3 / V-JEPA-2 "attentive probe" pattern: tokenise the
cached features, prepend one learnable query token, run a shallow bidirectional
transformer, and classify from the query's output embedding.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from . import images
from .geometry import FEATURE_NAMES
from .metric import DEFECTS

SEED = 0


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def _geometry_table(cache_dir: Path, split: str, item_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Load ``cache/geometry_<split>.csv`` aligned to ``item_ids`` + derived features."""
    df = pd.read_csv(cache_dir / f"geometry_{split}.csv")
    df = df.drop_duplicates("item_id").set_index("item_id").reindex(list(item_ids))
    g = df[list(FEATURE_NAMES)].astype(np.float32)
    # Scale-invariant derivations: triangle/edge size relative to the object size.
    diag = g["bbox_diag"].clip(lower=1e-12)
    g["log_faces_per_diag2"] = g["log_faces"] - 2.0 * np.log(diag)
    g["log_area_med_rel"] = np.log(g["area_med"].clip(lower=1e-30)) - 2.0 * np.log(diag)
    g["log_edge_med_rel"] = np.log(g["edge_len_med"].clip(lower=1e-30)) - np.log(diag)
    g["geom_missing"] = g["n_components"].isna().astype(np.float32)
    return g.to_numpy(dtype=np.float32), list(g.columns)


@dataclass
class Dataset:
    """Cached features (+ labels) for one split, aligned by ``item_ids`` order."""

    item_ids: np.ndarray
    grid: int  # pooled token grid per tile (always images.POOLED_GRID)
    dim: int
    tile_mean: torch.Tensor  # [n, 6, D] fp16, CPU
    tile_max: torch.Tensor
    tile_grid: torch.Tensor  # [n, 6, grid, grid, D] fp16, CPU
    geom: np.ndarray  # [n, F] float32, raw
    geom_names: list[str]
    labels: np.ndarray | None = None  # [n, 10] int8

    def __len__(self) -> int:
        return len(self.item_ids)

    def standardise(self, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Robust standardisation; NaN geometry (meshes above the feature cap) -> 0."""
        return standardise_geom(self.geom, median, scale)


def load_dataset(
    cache_dir: str | Path,
    data_dir: str | Path,
    model_key: str,
    split: str,
    limit: int | None = None,
) -> Dataset:
    cache_dir = Path(cache_dir)
    data_dir = Path(data_dir)
    cache = images.load_cache(cache_dir, model_key, split)
    if cache.grid != images.POOLED_GRID:
        raise ValueError(
            f"{images.cache_path(cache_dir, model_key, split)} stores a {cache.grid}x{cache.grid} grid; "
            f"this recipe needs {images.POOLED_GRID}x{images.POOLED_GRID} -- re-run `main.py features`"
        )
    item_ids = np.asarray(cache.item_ids)
    tile_mean, tile_max, tile_grid = cache.tile_mean, cache.tile_max, cache.tile_grid
    if limit is not None:
        item_ids, tile_mean, tile_max, tile_grid = (
            item_ids[:limit],
            tile_mean[:limit],
            tile_max[:limit],
            tile_grid[:limit],
        )
    geom, geom_names = _geometry_table(cache_dir, split, item_ids)

    labels = None
    if split == "train":
        lab = pd.read_csv(data_dir / "train.csv").set_index("item_id").reindex(list(item_ids))
        if lab[list(DEFECTS)].isna().any().any():
            raise ValueError("train labels missing for some cached items")
        labels = lab[list(DEFECTS)].to_numpy(dtype=np.int8)

    return Dataset(
        item_ids=item_ids,
        grid=cache.grid,
        dim=int(tile_mean.shape[-1]),
        tile_mean=torch.from_numpy(np.ascontiguousarray(tile_mean)),
        tile_max=torch.from_numpy(np.ascontiguousarray(tile_max)),
        tile_grid=torch.from_numpy(np.ascontiguousarray(tile_grid)),
        geom=geom,
        geom_names=geom_names,
        labels=labels,
    )


def standardise_geom(geom: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Robust standardisation; NaN geometry (meshes above the feature cap) -> 0."""
    z = (np.asarray(geom) - np.asarray(median)) / np.asarray(scale)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)


@dataclass
class GeometryTable:
    """Geometry-only features (+ labels): the fallback path needs no image cache."""

    item_ids: np.ndarray
    geom: np.ndarray  # [n, F] float32, raw
    geom_names: list[str]
    labels: np.ndarray | None = None  # [n, 10] int8

    def __len__(self) -> int:
        return len(self.item_ids)

    def standardise(self, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return standardise_geom(self.geom, median, scale)


def load_geometry(
    cache_dir: str | Path,
    data_dir: str | Path,
    split: str,
    limit: int | None = None,
) -> GeometryTable:
    """Geometry features for one split (no image caches), row order = ``data/<split>.csv``."""
    cache_dir, data_dir = Path(cache_dir), Path(data_dir)
    ids = pd.read_csv(data_dir / f"{split}.csv")["item_id"].to_numpy(dtype=str)
    if limit is not None:
        ids = ids[:limit]
    geom, geom_names = _geometry_table(cache_dir, split, ids)
    labels = None
    if split == "train":
        lab = pd.read_csv(data_dir / "train.csv").set_index("item_id").reindex(list(ids))
        if lab[list(DEFECTS)].isna().any().any():
            raise ValueError("train labels missing for some geometry rows")
        labels = lab[list(DEFECTS)].to_numpy(dtype=np.int8)
    return GeometryTable(item_ids=ids, geom=geom, geom_names=geom_names, labels=labels)


def geometry_stats(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust median / IQR scaling for the geometry block."""
    a = np.asarray(geom, dtype=np.float32)
    with np.errstate(all="ignore"):
        median = np.nan_to_num(np.nanmedian(a, axis=0), nan=0.0)
        q1, q3 = np.nanpercentile(a, [25, 75], axis=0)
    iqr = np.nan_to_num(q3 - q1, nan=0.0)
    scale = np.where(iqr > 1e-9, iqr, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class ProbeConfig:
    dim: int = 384  # DINOv3 feature width
    n_geom: int = 27  # set from data at train time (solution.py); keep in sync with _geometry_table
    grid: int = images.POOLED_GRID
    n_tiles: int = images.N_TILES
    d: int = 384
    depth: int = 4
    heads: int = 6
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainRecipe:
    """Canonical training defaults; the CLI and the ablation tool derive from here."""

    epochs: int = 40
    batch: int = 128
    lr: float = 1e-3
    wd: float = 0.05
    pos_weight_pow: float = 0.5
    seed: int = SEED
    folds: int = 5
    full_seeds: int = 1


class AttentiveProbe(nn.Module):
    """Shallow bidirectional transformer over cached patch tokens + one query."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        if cfg.grid != images.POOLED_GRID:
            raise ValueError(
                f"cfg.grid={cfg.grid} but the recipe tokenises {images.POOLED_GRID}x{images.POOLED_GRID} per tile"
            )
        self.cfg = cfg
        d = cfg.d
        self.proj_mean = nn.Linear(cfg.dim, d)
        self.proj_max = nn.Linear(cfg.dim, d)
        self.proj_grid = nn.Linear(cfg.dim, d)
        self.proj_geom = nn.Sequential(nn.Linear(cfg.n_geom, d), nn.GELU(), nn.Linear(d, d))
        self.tile_embed = nn.Embedding(cfg.n_tiles, d)
        self.type_embed = nn.Embedding(4, d)  # mean, max, grid, geom
        self.grid_pos = nn.Parameter(torch.zeros(1, cfg.grid * cfg.grid, d))
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(DEFECTS))

        for module in (self.proj_mean, self.proj_max, self.proj_grid, self.proj_geom):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p, std=0.02)
        for emb in (self.tile_embed, self.type_embed):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        nn.init.trunc_normal_(self.grid_pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.zeros_(self.head.bias)

    def tokens(self, mean, max_, grid_feats, geom):
        """Assemble the transformer input: [6 mean | 6 max | 6*g*g grid | GEOM | query].

        ``grid_feats`` is ``[b, 6, g, g, D]`` with g = ``images.POOLED_GRID``.
        """
        b = mean.shape[0]
        tiles = torch.arange(self.cfg.n_tiles, device=mean.device)
        grid_pos = self.grid_pos.view(1, self.cfg.grid * self.cfg.grid, self.cfg.d)
        # Project first, then add the tile identity: the raw feature width is
        # ``cfg.dim`` (384 for DINOv3-s, 768 for -b), the residual stream is ``d``.
        g = self.proj_grid(grid_feats.flatten(2, 3))  # [b, 6, g*g, d]
        g = g + self.tile_embed(tiles)[None, :, None]
        g = g + grid_pos[:, None]
        g = g + self.type_embed.weight[2]
        return torch.cat(
            [
                self.proj_mean(mean) + self.tile_embed(tiles)[None] + self.type_embed.weight[0],
                self.proj_max(max_) + self.tile_embed(tiles)[None] + self.type_embed.weight[1],
                g.flatten(1, 2),
                self.proj_geom(geom)[:, None] + self.type_embed.weight[3],
                self.query.expand(b, -1, -1),
            ],
            dim=1,
        )

    def forward(self, mean, max_, grid_feats, geom):
        x = self.blocks(self.tokens(mean, max_, grid_feats, geom))
        return self.head(self.norm(x[:, -1]))


# --------------------------------------------------------------------------- #
# training / inference
# --------------------------------------------------------------------------- #


def _tensors(ds: Dataset, idx: np.ndarray, geom_z: np.ndarray, device: str):
    # bf16 probe inputs on CUDA: the transformer runs under bf16 autocast anyway
    # and the cache is fp16, so bf16 loses no precision and halves the host copy.
    # On CPU there is no autocast, so keep fp32.  Geometry stays fp32.
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return (
        ds.tile_mean[idx].to(device, dtype, non_blocking=True),
        ds.tile_max[idx].to(device, dtype, non_blocking=True),
        ds.tile_grid[idx].to(device, dtype, non_blocking=True),
        torch.from_numpy(geom_z[idx]).to(device),
    )


@torch.inference_mode()
def predict_probs(
    model: AttentiveProbe,
    ds: Dataset,
    geom_z: np.ndarray,
    idx: np.ndarray,
    batch: int = 256,
    device: str | None = None,
) -> np.ndarray:
    device = device or str(next(model.parameters()).device)
    model.eval()
    out = np.empty((len(idx), len(DEFECTS)), dtype=np.float32)
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        with torch.autocast("cuda", torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits = model(*_tensors(ds, chunk, geom_z, device))
        out[i : i + len(chunk)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


@torch.inference_mode()
def _mean_loss(
    model: AttentiveProbe,
    ds: Dataset,
    geom_z: np.ndarray,
    idx: np.ndarray,
    batch: int = 256,
    device: str | None = None,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """Mean BCE on ``idx``; ``pos_weight`` makes it comparable to the training loss."""
    device = device or str(next(model.parameters()).device)
    y = torch.from_numpy(ds.labels[idx].astype(np.float32))
    total = 0.0
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        with torch.autocast("cuda", torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits = model(*_tensors(ds, chunk, geom_z, device))
        logits = logits.float().cpu()
        total += float(
            F.binary_cross_entropy_with_logits(
                logits, y[i : i + len(chunk)], pos_weight=None if pos_weight is None else pos_weight.cpu()
            )
        ) * len(chunk)
    return total / max(len(idx), 1)


def train_model(
    ds: Dataset,
    cfg: ProbeConfig,
    geom_z: np.ndarray,
    train_idx: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    wd: float = 0.05,
    batch: int = 128,
    pos_weight_pow: float = 0.5,
    seed: int = SEED,
    device: str | None = None,
    log_every: int = 0,
    val_idx: np.ndarray | None = None,
    val_history: list[float] | None = None,
    amp: bool = True,
) -> tuple[AttentiveProbe, list[float]]:
    """Train one probe on ``train_idx``; ``val_idx`` is scored every epoch.

    ``val_history`` (optional list) receives the per-epoch validation loss, which
    is what the presentation's learning curves plot; the training loop itself
    never uses it to make decisions.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_amp = amp and device.startswith("cuda")

    y = ds.labels
    assert y is not None, "training requires labels"
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    pos = torch.from_numpy(y[train_idx].sum(0).astype(np.float32))
    neg = len(train_idx) - pos
    pos_weight = torch.clamp((neg / pos.clamp(min=1.0)) ** pos_weight_pow, 1.0, 20.0).to(device)

    model = AttentiveProbe(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, int(np.ceil(len(train_idx) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1, div_factor=10.0, final_div_factor=100.0
    )
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(*_tensors(ds, idx, geom_z, device))
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        history.append(total / len(perm))
        val_loss = None
        if val_idx is not None:
            val_loss = _mean_loss(model, ds, geom_z, val_idx, device=device, pos_weight=pos_weight)
            if val_history is not None:
                val_history.append(val_loss)
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            msg = f"    epoch {epoch + 1}/{epochs} loss {history[-1]:.4f}"
            if val_loss is not None:
                msg += f" val {val_loss:.4f}"
            print(msg, flush=True)
    return model, history


def folds(n: int, k: int = 5, seed: int = SEED) -> list[np.ndarray]:
    """Index folds; the labels are multi-label, so plain shuffled K-fold is used."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def histgb_fit(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    max_components: int | None = None,
) -> tuple[list, tuple | None]:
    """Fit one HistGradientBoostingClassifier per label (optional PCA front-end)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    scaler = pca = None
    if max_components is not None and X.shape[1] > max_components:
        scaler = StandardScaler().fit(X)
        pca = PCA(n_components=max_components, random_state=seed).fit(scaler.transform(X))
        X = pca.transform(scaler.transform(X))
    models = []
    for j in range(y.shape[1]):
        clf = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
        clf.fit(X, y[:, j])
        models.append(clf)
    return models, ((scaler, pca) if pca is not None else None)


def histgb_oof(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    seed: int = SEED,
    max_components: int | None = None,
) -> np.ndarray:
    """Cheap 5-fold gradient-boosting reference on a feature block (optional PCA)."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    oof = np.zeros((len(X), y.shape[1]), dtype=np.float32)
    for val in folds(len(X), k, seed):
        train = np.setdiff1d(np.arange(len(X)), val)
        models, pre = histgb_fit(X[train], y[train], seed=seed, max_components=max_components)
        Xva = X[val]
        if pre is not None:
            scaler, pca = pre
            Xva = pca.transform(scaler.transform(Xva))
        for j, clf in enumerate(models):
            oof[val, j] = clf.predict_proba(Xva)[:, 1]
    return oof


def run_cv(
    ds: Dataset,
    cfg: ProbeConfig,
    geom_z: np.ndarray,
    *,
    k: int = 5,
    fold_seed: int = SEED,
    model_seed: int = SEED,
    only_folds: Iterable[int] | None = None,
    log=print,
    **train_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated OOF probabilities + the mask of evaluated rows.

    ``fold_seed`` fixes the split, ``model_seed`` the per-fold initialisation
    (``model_seed + f``); ``only_folds`` runs a subset of the folds, which is how
    the ablation tool screens a single fold.  ``train_kwargs`` go to
    :func:`train_model`.
    """
    n = len(ds)
    idx = np.arange(n)
    oof = np.zeros((n, len(DEFECTS)), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    split = folds(n, k, fold_seed)
    for f in range(k) if only_folds is None else only_folds:
        val = split[f]
        train = np.setdiff1d(idx, val, assume_unique=False)
        t0 = time.time()
        model, hist = train_model(ds, cfg, geom_z, train, seed=model_seed + f, **train_kwargs)
        oof[val] = predict_probs(model, ds, geom_z, val)
        done[val] = True
        log(f"  fold {f + 1}/{k}: loss {hist[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return oof, done
