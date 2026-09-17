"""CPU tests for the mesh-quality pipeline seams.

Everything here runs in seconds without a GPU, real data or the DINOv3
backbone: the extraction test swaps in a deterministic stub model, and the
fusion test builds a tiny synthetic cache.  These pin the interfaces the
refactor leans on (cache layout, token order, metric identity, threshold
tuning, fusion reproducibility).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mesh_quality import images, metric, model, solution

DEFECTS = list(metric.DEFECTS)


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #


def write_cache(cache_dir: Path, key: str, split: str, n: int, dim: int = 4, ids=None) -> np.ndarray:
    """Write a minimal but valid pooled cache directory; returns its item ids."""
    rng = np.random.default_rng(0)
    if ids is None:
        ids = np.array([f"item{i:03d}" for i in range(n)])
    dest = images.cache_path(cache_dir, key, split)
    dest.mkdir(parents=True, exist_ok=True)
    for name, shape in (
        ("tile_mean", (n, images.N_TILES, dim)),
        ("tile_max", (n, images.N_TILES, dim)),
        ("tile_grid", (n, images.N_TILES, images.POOLED_GRID, images.POOLED_GRID, dim)),
    ):
        np.save(dest / f"{name}.npy", rng.standard_normal(shape).astype(np.float16))
    np.save(dest / "item_ids.npy", np.asarray(ids))
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "model_key": key,
                "model": images.MODELS[key],
                "split": split,
                "grid_size": images.POOLED_GRID,
                "native_grid_size": images.GRID,
                "dim": dim,
                "n_tiles": images.N_TILES,
                "n_items": n,
                "dtype": "float16",
            }
        )
    )
    return np.asarray(ids)


def write_geometry(cache_dir: Path, split: str, ids) -> None:
    from mesh_quality.geometry import FEATURE_NAMES

    df = pd.DataFrame({"item_id": list(ids)})
    for name in FEATURE_NAMES:
        df[name] = np.linspace(0.0, 1.0, len(ids))
    df.to_csv(cache_dir / f"geometry_{split}.csv", index=False)


def write_labels(data_dir: Path, ids, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = (rng.random((len(ids), len(DEFECTS))) < 0.3).astype(np.int8)
    df = pd.DataFrame({"item_id": list(ids)})
    for k, name in enumerate(DEFECTS):
        df[name] = labels[:, k]
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "train.csv", index=False)
    return labels


# --------------------------------------------------------------------------- #
# pooling / extraction
# --------------------------------------------------------------------------- #


def test_pool_patches_is_block_mean():
    x = torch.arange(1024, dtype=torch.float32).reshape(1, images.GRID, images.GRID, 1)
    out = images.pool_patches(x)
    assert out.shape == (1, images.POOLED_GRID, images.POOLED_GRID, 1)
    assert out[0, 0, 0, 0].item() == 49.5  # mean of rows 0-3 x cols 0-3 of arange(1024)
    z = torch.randn(3, images.GRID, images.GRID, 5)
    assert torch.allclose(
        images.pool_patches(z), z.reshape(3, 8, 4, 8, 4, 5).mean(dim=(2, 4)), atol=1e-6
    )
    with pytest.raises(ValueError):
        images.pool_patches(torch.randn(2, 4, 4, 3))


class _StubBackbone(torch.nn.Module):
    """Deterministic DINOv3 stand-in: patch token channel c = patch mean * (c+1)."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.embed_dim = dim
        self.num_prefix_tokens = 1

    def forward_features(self, x):
        b = x.shape[0]
        base = x.reshape(b, 3, 32, 16, 32, 16).mean(dim=(1, 3, 5))  # [b, 32, 32] normalised grey
        patches = base.unsqueeze(-1) * torch.arange(1, self.embed_dim + 1).float()
        patches = patches.reshape(b, 1024, self.embed_dim)
        return torch.cat([torch.zeros(b, 1, self.embed_dim), patches], dim=1)


def _stub_expectation(path: Path, dim: int = 4):
    """Independently recompute the stub features: (tile_mean, tile_max, pooled grid)."""
    tiles = images.split_tiles(path).astype(np.float32) / 255.0
    x = (tiles - np.array(images.IMAGENET_MEAN, dtype=np.float32)) / np.array(images.IMAGENET_STD, dtype=np.float32)
    base = x.reshape(images.N_TILES, 32, 16, 32, 16, 3).mean(axis=(2, 4, 5))  # [6, 32, 32]
    patches = base[..., None] * np.arange(1, dim + 1, dtype=np.float32)  # [6, 32, 32, dim]
    flat = patches.reshape(images.N_TILES, 1024, dim)
    pooled = patches.reshape(images.N_TILES, 8, 4, 8, 4, dim).mean(axis=(2, 4))
    return flat.mean(1), flat.max(1), pooled


def _tiny_dataset(tmp_path: Path) -> tuple[Path, list[str]]:
    data = tmp_path / "data"
    (data / "test").mkdir(parents=True)
    ids = ["a", "b"]
    pd.DataFrame({"item_id": ids}).to_csv(data / "test.csv", index=False)
    rng = np.random.default_rng(0)
    for item in ids:
        Image.fromarray(rng.integers(0, 256, (1024, 1536, 3), dtype=np.uint8)).save(data / "test" / f"{item}.png")
    return data, ids


def test_extract_roundtrip(tmp_path, monkeypatch):
    data, ids = _tiny_dataset(tmp_path)
    stub = _StubBackbone(dim=4)
    monkeypatch.setattr(images, "_load_model", lambda key, device: stub)

    images.extract_split(data, "test", tmp_path / "cache", model_key="s", batch_tiles=6, device="cpu")
    cache = images.load_cache(tmp_path / "cache", "s", "test")
    assert list(cache.item_ids) == ids
    assert cache.grid == images.POOLED_GRID and cache.dim == 4
    assert cache.tile_grid.shape == (2, 6, 8, 8, 4)
    for i, item in enumerate(ids):
        exp_mean, exp_max, exp_grid = _stub_expectation(data / "test" / f"{item}.png")
        assert np.allclose(cache.tile_mean[i], exp_mean, atol=0.02)
        assert np.allclose(cache.tile_max[i], exp_max, atol=0.02)
        assert np.allclose(cache.tile_grid[i], exp_grid, atol=0.02)


def test_extract_batching_is_deterministic(tmp_path, monkeypatch):
    data, _ = _tiny_dataset(tmp_path)
    monkeypatch.setattr(images, "_load_model", lambda key, device: _StubBackbone(dim=4))
    images.extract_split(data, "test", tmp_path / "c1", model_key="s", batch_tiles=6, device="cpu")
    images.extract_split(data, "test", tmp_path / "c2", model_key="s", batch_tiles=12, device="cpu")
    a = images.load_cache(tmp_path / "c1", "s", "test")
    b = images.load_cache(tmp_path / "c2", "s", "test")
    for name in ("tile_mean", "tile_max", "tile_grid"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_load_dataset_alignment(tmp_path):
    ids = write_cache(tmp_path / "cache", "s", "train", n=3)
    write_geometry(tmp_path / "cache", "train", ids)
    labels = write_labels(tmp_path / "data", ids)

    ds = model.load_dataset(tmp_path / "cache", tmp_path / "data", "s", "train")
    assert list(ds.item_ids) == list(ids)
    assert ds.grid == images.POOLED_GRID
    assert ds.tile_grid.shape == (3, 6, 8, 8, 4)
    assert np.array_equal(ds.labels, labels)
    assert ds.standardise(np.zeros(27), np.ones(27)).shape == (3, 27)

    ds2 = model.load_dataset(tmp_path / "cache", tmp_path / "data", "s", "train", limit=2)
    assert list(ds2.item_ids) == list(ids[:2])


def test_load_dataset_rejects_native_grid(tmp_path):
    write_cache(tmp_path / "cache", "s", "train", n=2)
    meta = tmp_path / "cache" / "dinos_train" / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "grid_size": images.GRID}))
    with pytest.raises(ValueError, match="re-run `main.py features`"):
        model.load_dataset(tmp_path / "cache", tmp_path / "data", "s", "train")


# --------------------------------------------------------------------------- #
# geometry table
# --------------------------------------------------------------------------- #


def test_load_geometry_uses_data_order(tmp_path):
    cache, data = tmp_path / "cache", tmp_path / "data"
    cache.mkdir()
    ids = ["b", "a"]
    write_geometry(cache, "train", ["a", "b"])
    write_labels(data, ids)
    table = model.load_geometry(cache, data, "train")
    assert list(table.item_ids) == ids  # data/train.csv order, not the geometry csv order
    assert table.geom.shape == (2, 27)
    assert table.labels is not None and table.labels.shape == (2, 10)


def test_standardise_geom_handles_nan_and_clips():
    geom = np.array([[1.0, np.nan, np.inf, 100.0]])
    z = model.standardise_geom(geom, np.zeros(4), np.ones(4))
    assert z[0, 1] == 0.0 and z[0, 2] == 0.0 and z[0, 3] == 8.0


# --------------------------------------------------------------------------- #
# probe token layout
# --------------------------------------------------------------------------- #


def test_token_layout_and_count():
    cfg = model.ProbeConfig(dim=4, n_geom=3, grid=images.POOLED_GRID, d=8, depth=1, heads=2)
    net = model.AttentiveProbe(cfg)
    mean, mx = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    grid, geom = torch.randn(2, 6, 8, 8, 4), torch.randn(2, 3)

    tokens = net.tokens(mean, mx, grid, geom)
    assert tokens.shape == (2, 398, 8)  # 6 mean + 6 max + 384 grid + geom + query

    def changed(other) -> list[int]:
        with torch.no_grad():
            diff = (tokens != net.tokens(*other)).any(-1).nonzero(as_tuple=True)[1]
        return sorted(set(diff.tolist()))

    assert changed((mean + 1, mx, grid, geom)) == list(range(6))
    assert changed((mean, mx + 1, grid, geom)) == list(range(6, 12))
    assert changed((mean, mx, grid + 1, geom)) == list(range(12, 396))
    assert changed((mean, mx, grid, geom + 1)) == [396]  # query (397) never changes


def test_probe_rejects_other_grid():
    cfg = model.ProbeConfig(dim=4, n_geom=3, grid=16, d=8, depth=1, heads=2)
    with pytest.raises(ValueError, match="tokenises"):
        model.AttentiveProbe(cfg)


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #


def test_derive_quality_identity():
    y = np.zeros((3, 10), dtype=np.int8)
    y[1, 3] = 1
    y[2, :] = 1
    assert metric.derive_quality(y).tolist() == [1, 0, 0]


def test_tune_thresholds_deterministic_and_at_least_baseline():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 10)) < 0.2).astype(np.int8)
    probs = np.clip(y * 0.5 + rng.random((200, 10)) * 0.6, 0, 1).astype(np.float32)
    yq = metric.derive_quality(y)
    thr1, res1 = metric.tune_thresholds(probs, y, yq)
    thr2, res2 = metric.tune_thresholds(probs, y, yq)
    assert np.array_equal(thr1, thr2) and res1["score"] == res2["score"]
    d = (probs >= 0.5).astype(np.int8)
    base = metric.score_predictions(y, yq, d, metric.derive_quality(d))
    assert res1["score"] >= base["score"] - 1e-9


def test_submission_roundtrip(tmp_path):
    ids = ["a", "b"]
    d_pred = np.zeros((2, 10), dtype=np.int8)
    d_pred[0, 0] = 1
    path = metric.write_submission(tmp_path / "sub.csv", ids, d_pred)
    back = metric.read_submission(path)
    assert back["item_id"].tolist() == ids
    assert back[list(metric.DEFECTS)].to_numpy(dtype=np.int8).tolist() == d_pred.tolist()
    assert back["quality"].tolist() == [0, 1]  # derived: row 0 has a defect


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #


def _fuse_fixture(tmp_path: Path, n: int = 8):
    cache = tmp_path / "cache"
    ids = write_cache(cache, "s", "train", n=n)
    write_cache(cache, "b", "train", n=n)
    write_geometry(cache, "train", ids)
    labels = write_labels(tmp_path / "data", ids)
    rng = np.random.default_rng(1)
    oof = {k: rng.random((n, 10)).astype(np.float32) for k in ("s", "b")}
    for k, matrix in oof.items():
        np.save(cache / f"oof_probe_{k}.npy", matrix)
        np.save(cache / f"probs_test_{k}.npy", rng.random((n, 10)).astype(np.float32))
    args = argparse.Namespace(task_dir=tmp_path, data_dir=tmp_path / "data", models="s,b", weights=None)
    return args, cache, labels, oof


def test_fuse_reproducible(tmp_path):
    args, cache, labels, oof = _fuse_fixture(tmp_path)
    solution.fuse(args)
    payload = json.loads((cache / "fuse_s+b.json").read_text())
    assert payload["models"] == ["s", "b"] and payload["weights"] == [0.5, 0.5]
    avg = 0.5 * oof["s"] + 0.5 * oof["b"]
    thr, _ = metric.tune_thresholds(avg, labels, metric.derive_quality(labels))
    assert np.allclose(payload["thresholds"], thr)
    solution.fuse(args)  # same inputs -> same thresholds
    assert json.loads((cache / "fuse_s+b.json").read_text())["thresholds"] == payload["thresholds"]


def test_fuse_rejects_mismatched_item_order(tmp_path):
    args, cache, _, _ = _fuse_fixture(tmp_path)
    write_cache(cache, "b", "train", n=8, ids=[f"other{i}" for i in range(8)])
    with pytest.raises(SystemExit, match="train item order differs"):
        solution.fuse(args)
