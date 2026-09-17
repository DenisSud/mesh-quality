"""CPU smoke tests for the mesh-patch JEPA encoder.

Seconds, no GPU, no real data: synthetic caches and a tiny model. These pin the
seams the training run depends on (SIGReg anti-collapse behaviour, mask
invariants, view-only steps, checkpoint round-trip, embedding dump shapes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mesh_quality import jepa, mesh_tokens as mt

K, KP, F, D = 256, 32, len(mt.POINT_NAMES), len(mt.STAT_NAMES)


def tiny_cfg(**kw) -> jepa.JepaConfig:
    base = dict(d=32, depth=2, heads=2, pred_d=24, pred_depth=2, pred_heads=2, batch=4,
                sigreg_slices=32, sigreg_knots=9, point_hidden=16, amp=False)
    base.update(kw)
    return jepa.JepaConfig(**base)


def fake_batch(b: int = 4, n_valid: int = K, seed: int = 0, empty_item: bool = False) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    valid = torch.zeros(b, K, dtype=torch.bool)
    valid[:, :n_valid] = True
    if empty_item:
        valid[0] = False
    pts = torch.randn(b, K, KP, F, generator=gen) * 0.5
    stats = torch.randn(b, K, D, generator=gen)
    centers = torch.rand(b, K, 3, generator=gen) - 0.5
    return {"pts": pts.half(), "stats": stats, "centers": centers, "valid": valid,
            "globals": torch.zeros(b, len(mt.GLOBAL_NAMES))}


# --------------------------------------------------------------------------- #
# SIGReg
# --------------------------------------------------------------------------- #


def test_sigreg_low_for_gaussian_high_for_collapse():
    gen = torch.Generator().manual_seed(0)
    gauss = torch.randn(4096, 64, generator=gen)
    collapsed = torch.zeros(4096, 64) + 0.01
    shifted = torch.randn(4096, 64, generator=gen) * 3.0
    assert jepa.sigreg(gauss, slices=64, knots=9) < 0.02
    assert jepa.sigreg(collapsed, slices=64, knots=9) > 10 * jepa.sigreg(gauss, slices=64, knots=9)
    assert jepa.sigreg(shifted, slices=64, knots=9) > 5 * jepa.sigreg(gauss, slices=64, knots=9)


def test_sigreg_backward():
    z = torch.randn(64, 16, requires_grad=True)
    jepa.sigreg(z, slices=16, knots=7).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# masking invariants
# --------------------------------------------------------------------------- #


def test_mask_is_spatial_subset_of_valid_with_context():
    model = jepa.MeshJepa(tiny_cfg(mask_probability=1.0))
    batch = fake_batch(b=8, n_valid=64)
    gen = torch.Generator().manual_seed(3)
    mask = model.sample_mask(batch["centers"], batch["valid"], gen)
    assert not (mask & ~batch["valid"]).any(), "mask must not touch invalid patches"
    counts = mask.sum(1).float()
    assert counts.min() > 0, "at least one patch masked on active items"
    assert (counts <= 0.45 * 64 + 1).all(), "mask ratio above the configured range"
    ctx = (batch["valid"] & ~mask).sum(1)
    assert ctx.min() >= 3, "masking must leave context to predict from"


def test_mask_probability_leaves_view_only_steps():
    model = jepa.MeshJepa(tiny_cfg(mask_probability=0.25))
    batch = fake_batch(b=64, n_valid=64)
    mask = model.sample_mask(batch["centers"], batch["valid"], torch.Generator().manual_seed(5))
    frac = float((mask.sum(1) > 0).float().mean())
    assert 0.05 < frac < 0.5, f"~25% of items should be view-only, got masked fraction {frac:.2f}"


def test_mask_over_blocks_not_uniform():
    """Spatial blocks: the masked set should be more clustered than random."""
    model = jepa.MeshJepa(tiny_cfg(mask_lo=0.15, mask_hi=0.15, mask_blocks=(1, 1), mask_probability=1.0))
    batch = fake_batch(b=4, n_valid=K, seed=7)
    gen = torch.Generator().manual_seed(11)
    mask = model.sample_mask(batch["centers"], batch["valid"], gen)
    c = batch["centers"]
    seen = 0
    for i in range(len(mask)):
        idx = torch.nonzero(mask[i], as_tuple=False).squeeze(1)
        if len(idx) == 0:
            continue
        seen += 1
        assert len(idx) > 8
        inside = torch.cdist(c[i][idx], c[i][idx]).mean()
        assert inside < 0.8 * torch.cdist(c[i], c[i]).mean(), "mask should be spatially coherent"
    assert seen >= 3


def test_tiny_mesh_still_trains():
    model = jepa.MeshJepa(tiny_cfg())
    batch = fake_batch(b=3, n_valid=4, empty_item=True)   # 4-face object + a fully empty item
    out = model(batch, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# --------------------------------------------------------------------------- #
# forward / backward
# --------------------------------------------------------------------------- #


def test_forward_backward_and_components():
    cfg = tiny_cfg()
    model = jepa.MeshJepa(cfg)
    model.set_feature_norm({
        "point_mean": torch.zeros(len(mt.POINT_NAMES)), "point_std": torch.ones(len(mt.POINT_NAMES)),
        "stat_mean": torch.zeros(D), "stat_std": torch.ones(D),
        "global_mean": torch.zeros(len(mt.GLOBAL_NAMES)), "global_std": torch.ones(len(mt.GLOBAL_NAMES)),
    })
    batch = fake_batch(b=4)
    out = model(batch, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert 0.0 < float(out["masked_frac"]) < 0.6
    assert torch.isfinite(out["pred"]) and torch.isfinite(out["sigreg"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.tokenizer.point[0].weight.grad is not None
    assert model.predictor.layers[0].linear1.weight.grad is not None


def test_view_only_step_when_no_mask():
    cfg = tiny_cfg(mask_probability=0.0)
    model = jepa.MeshJepa(cfg)
    out = model(fake_batch(), generator=torch.Generator().manual_seed(0))
    assert float(out["masked_frac"]) == 0.0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


# --------------------------------------------------------------------------- #
# checkpoint / embedding dump
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip_and_dump(tmp_path: Path):
    cfg = tiny_cfg(point_dropout=0.0, point_jitter=0.0, patch_dropout=0.0)
    model = jepa.MeshJepa(cfg)
    model.set_feature_norm({
        "point_mean": torch.zeros(len(mt.POINT_NAMES)), "point_std": torch.ones(len(mt.POINT_NAMES)),
        "stat_mean": torch.zeros(D), "stat_std": torch.ones(D),
        "global_mean": torch.zeros(len(mt.GLOBAL_NAMES)), "global_std": torch.ones(len(mt.GLOBAL_NAMES)),
    })
    path = tmp_path / "ckpt.pt"
    jepa.save_checkpoint(model, cfg, path, epoch=3, log=[])
    loaded, cfg2 = jepa.load_checkpoint(path)
    assert cfg2.d == cfg.d
    batch = fake_batch()
    with torch.no_grad():
        a = model(batch, generator=torch.Generator().manual_seed(0))["pooled"]
        b = loaded(batch, generator=torch.Generator().manual_seed(0))["pooled"]
    assert torch.allclose(a, b, atol=1e-5)


def test_synthetic_cache_and_embeddings(tmp_path: Path):
    cfg = mt.TokenConfig(n_sample=64, n_patch=8, k_points=4, voxel_res=4)
    n = 5
    cache = tmp_path / "cache" / "train"
    cache.mkdir(parents=True)
    shapes = {
        "pts": ((n, cfg.n_patch, cfg.k_points, cfg.n_feat), np.float16),
        "stats": ((n, cfg.n_patch, cfg.n_stats), np.float16),
        "centers": ((n, cfg.n_patch, 3), np.float16),
        "radius": ((n, cfg.n_patch), np.float16),
        "voxel": ((n, cfg.voxel_res**3), np.float16),
        "globals": ((n, cfg.n_global), np.float32),
        "valid": ((n, cfg.n_patch), np.uint8),
    }
    for name, (shape, dtype) in shapes.items():
        np.save(cache / f"{name}.npy", np.zeros(shape, dtype))
    (cache / "item_ids.txt").write_text("\n".join(f"id{i}" for i in range(n)) + "\n")
    valid = np.load(cache / "valid.npy")
    valid[:, :4] = 1
    np.save(cache / "valid.npy", valid)

    ds = jepa.MeshPatchDataset(tmp_path / "cache", "train")
    assert len(ds) == n and ds[0]["pts"].shape == (cfg.n_patch, cfg.k_points, cfg.n_feat)
    model = jepa.MeshJepa(tiny_cfg())
    model.set_feature_norm(jepa.compute_feature_stats(ds, n=2))
    emb = jepa.extract_embeddings(model, ds, torch.device("cpu"), batch=2)
    assert emb["tokens"].shape == (n, cfg.n_patch, tiny_cfg().d)
    assert emb["tokens"].dtype == np.float16
    assert emb["pooled"].shape == (n, tiny_cfg().d)
    assert np.isfinite(emb["pooled"]).all()


def test_multi_split_dataset(tmp_path: Path):
    cfg = mt.TokenConfig(n_sample=64, n_patch=8, k_points=4, voxel_res=4)
    root = tmp_path / "cache"
    for split, n in (("train", 3), ("test", 2)):
        d = root / split
        d.mkdir(parents=True)
        for name, shape, dtype in (
            ("pts", (n, cfg.n_patch, cfg.k_points, cfg.n_feat), np.float16),
            ("stats", (n, cfg.n_patch, cfg.n_stats), np.float16),
            ("centers", (n, cfg.n_patch, 3), np.float16),
            ("radius", (n, cfg.n_patch), np.float16),
            ("voxel", (n, cfg.voxel_res**3), np.float16),
            ("globals", (n, cfg.n_global), np.float32),
            ("valid", (n, cfg.n_patch), np.uint8),
        ):
            np.save(d / f"{name}.npy", np.zeros(shape, dtype))
        (d / "item_ids.txt").write_text("\n".join(f"{split}{i}" for i in range(n)) + "\n")
    ds = jepa.make_dataset(root, "both")
    assert len(ds) == 5
    assert ds.item_ids == ["train0", "train1", "train2", "test0", "test1"]
    assert ds[3]["index"] == 3 and ds[4]["pts"].shape == (cfg.n_patch, cfg.k_points, cfg.n_feat)
    stats = jepa.compute_feature_stats(ds, n=4)
    assert stats["point_std"].shape == (cfg.n_feat,)
