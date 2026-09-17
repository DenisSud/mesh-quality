"""Tests for the L3 mesh-token probe tool (tools/probe_mesh.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "tools"))

import probe_mesh as P  # noqa: E402
from mesh_quality import model as M  # noqa: E402


def tiny_cfg() -> M.ProbeConfig:
    return M.ProbeConfig(n_geom=5, d=32, depth=1, heads=2, dropout=0.0)


def base_tensors(b: int = 2, dim: int = 384, grid: int = 8, n_geom: int = 5):
    torch.manual_seed(0)
    return (
        torch.randn(b, 6, dim),
        torch.randn(b, 6, dim),
        torch.randn(b, 6, grid, grid, dim),
        torch.randn(b, n_geom),
    )


def test_mesh_tokens_are_appended_after_the_query():
    cfg = tiny_cfg()
    probe = P.MeshTokenProbe(cfg, mesh_dim=8).eval()
    base = base_tensors()
    with torch.no_grad():
        plain = probe(*base)
        mesh = (torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool))
        with_mesh = probe(*base, mesh=mesh)
        tokens = probe.tokens(*base, mesh=mesh)
    n_base = 6 + 6 + 6 * cfg.grid * cfg.grid + 1 + 1
    assert plain.shape == (2, len(M.DEFECTS))
    assert with_mesh.shape == plain.shape
    assert tokens.shape == (2, n_base + 4, cfg.d)


def test_invalid_patches_get_the_null_embedding():
    cfg = tiny_cfg()
    probe = P.MeshTokenProbe(cfg, mesh_dim=8).eval()
    base = base_tensors(b=1)
    emb = torch.randn(1, 3, 8)
    with torch.no_grad():
        all_valid = probe.tokens(*base, mesh=(emb, torch.ones(1, 3, dtype=torch.bool)))
        one_bad = probe.tokens(*base, mesh=(emb, torch.tensor([[True, True, False]])))
    assert not torch.allclose(all_valid, one_bad)  # the validity flag is visible to the probe


def test_mesh_probe_context_swaps_and_restores_the_shipped_globals():
    before = (M._tensors, M.AttentiveProbe)
    with P.mesh_probe(mesh_dim=16):
        assert M._tensors is not before[0]
        assert M.AttentiveProbe is not before[1]
        assert isinstance(M.AttentiveProbe(tiny_cfg()), P.MeshTokenProbe)
    assert (M._tensors, M.AttentiveProbe) == before


def test_load_mesh_block_realigns_and_flags_missing_items(tmp_path: Path):
    cache = tmp_path / "cache" / "mesh_tokens" / "train"
    cache.mkdir(parents=True)
    (cache / "item_ids.txt").write_text("b\na\nc\n")
    tokens = np.arange(3 * 2 * 4, dtype=np.float16).reshape(3, 2, 4)
    np.save(tmp_path / "t_tokens.npy", tokens)
    np.save(tmp_path / "t_valid.npy", np.ones((3, 2), bool))

    probe_ids = np.array(["c", "missing", "a"])
    got, valid = P.load_mesh_block(tmp_path / "t", "train", probe_ids, tmp_path / "cache")
    assert torch.equal(got[0], torch.from_numpy(tokens[2]))
    assert torch.equal(got[2], torch.from_numpy(tokens[1]))
    assert not valid[1].any()  # item with no tokeniser row stays invalid
    assert bool(valid[[0, 2]].all())


def test_tokeniser_order_maps_and_flags_missing(tmp_path: Path):
    cache = tmp_path / "cache" / "mesh_tokens" / "train"
    cache.mkdir(parents=True)
    (cache / "item_ids.txt").write_text("b\na\nc\n")
    order = P.tokeniser_order(tmp_path / "unused", "train", np.array(["c", "gone", "a"]), tmp_path / "cache")
    assert order.tolist() == [2, -1, 1]


class _FakeBatcher:
    def __init__(self) -> None:
        self.got: np.ndarray | None = None

    def gather(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        self.got = np.asarray(rows)
        n = len(rows)
        return {
            "pts": torch.zeros(n, 2, 3, 11, dtype=torch.float16),
            "stats": torch.zeros(n, 2, 16),
            "centers": torch.zeros(n, 2, 3),
            "valid": torch.ones(n, 2, dtype=torch.bool),
            "globals": torch.zeros(n, 11),
        }


def test_raw_batch_clamps_missing_rows_and_marks_them_invalid():
    batcher = _FakeBatcher()
    order = np.array([5, -1, 7])
    ds = P.RawBatchDataset(object(), batcher, order, valid=torch.tensor([True, False, True]))
    raw = ds.raw_batch(np.array([0, 1, 2]), "cpu")
    assert batcher.got.tolist() == [5, 0, 7]  # -1 clamped: no wrap-around into the last row
    assert not raw[3][1].any()  # the skipped item contributes no valid patch
    assert bool(raw[3][[0, 2]].all())


def test_joint_factory_builds_an_independent_encoder_per_fold():
    from mesh_quality import jepa as J

    jcfg = J.JepaConfig(d=32, depth=1, heads=2, point_hidden=16, pred_d=32, pred_depth=1)
    base = J.MeshJepa(jcfg)
    factory = P.make_joint_factory(base.state_dict(), jcfg)
    fold_a, fold_b = factory(tiny_cfg()), factory(tiny_cfg())
    assert fold_a.encoder is not fold_b.encoder
    before = next(fold_b.encoder.parameters()).detach().clone()
    with torch.no_grad():
        for p in fold_a.encoder.parameters():
            p.add_(1.0)  # simulate fold A's optimiser step
    assert torch.allclose(next(fold_b.encoder.parameters()), before), (
        "fold B inherited fold A's encoder weights -- CV would be leaked"
    )


def test_joint_probe_pushes_gradients_into_the_encoder():
    from mesh_quality import jepa as J
    from mesh_quality import mesh_tokens as mt

    enc = J.MeshJepa(J.JepaConfig(d=32, depth=1, heads=2, point_hidden=16, pred_d=32, pred_depth=1))
    probe = P.JointMeshProbe(tiny_cfg(), encoder=enc)
    base = base_tensors(b=2, grid=8)
    raw = (
        torch.randn(2, 4, 8, 11) * 0.1,
        torch.randn(2, 4, 16),
        torch.randn(2, 4, 3) * 0.1,
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
        torch.randn(2, len(mt.GLOBAL_NAMES)) * 0.1,
    )
    probe(*base, raw=raw).sum().backward()
    encoder_grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert encoder_grads, "no gradients reached the encoder -- L4 would silently be L3"
    assert any(float(g.abs().sum()) > 0 for g in encoder_grads)
