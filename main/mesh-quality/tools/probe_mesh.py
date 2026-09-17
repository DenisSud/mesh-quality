"""L3 gate: append frozen mesh-JEPA patch tokens to the shipped attentive probe.

The shipped recipe is reused verbatim: ``model.run_cv`` -> ``train_model`` look up
two module globals (``_tensors`` and ``AttentiveProbe``), so this tool swaps both
inside a context manager instead of touching ``model.py``.  The shipped probe
path therefore stays byte-identical; anything here is opt-in.

    devenv shell -- uv run python main/mesh-quality/tools/probe_mesh.py \
        --dump-stem main/mesh-quality/cache/jepa_pretrain/run1/e_train --only-folds 0

Reference rows scored with the same threshold tuning: ``cache/oof_probe_s.npy``
(shipped attentive probe, OOF 13.68) and ``cache/oof_histgb_screen_geometry.npy``
(geometry-only HistGB, 10.52).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import jepa  # noqa: E402
from mesh_quality import mesh_tokens as mt  # noqa: E402
from mesh_quality import metric, model as M  # noqa: E402

_base_tensors = M._tensors  # captured before any patching


# --------------------------------------------------------------------------- #
# mesh token block
# --------------------------------------------------------------------------- #


def load_mesh_block(dump_stem: Path, split: str, item_ids: np.ndarray, cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Dumped JEPA patch tokens, reordered onto the probe's row order.

    Items the tokeniser skipped (no structure) keep zero tokens with ``valid=False``.
    """
    stem = Path(dump_stem)
    tokens = np.load(stem.with_name(stem.name + "_tokens.npy"))  # [n_tok, K, d] fp16
    valid = np.load(stem.with_name(stem.name + "_valid.npy"))  # [n_tok, K] bool
    ids_path = cache_dir / "mesh_tokens" / split / "item_ids.txt"
    tok_ids = [line.strip() for line in ids_path.read_text().splitlines()]
    if len(tok_ids) != len(tokens):
        raise SystemExit(f"{stem}: {len(tokens)} token rows but {len(tok_ids)} item ids in {ids_path}")
    row = {item: i for i, item in enumerate(tok_ids)}
    order = np.array([row.get(item, -1) for item in item_ids], dtype=np.int64)
    missing = int((order < 0).sum())
    filled = np.where(order >= 0, order, 0)
    aligned = torch.from_numpy(np.ascontiguousarray(tokens[filled]))
    aligned_valid = torch.from_numpy(np.ascontiguousarray(valid[filled]))
    aligned_valid[order < 0] = False
    print(
        f"  mesh block: {len(item_ids)} rows from {stem.name} "
        f"({missing} missing, valid patches {aligned_valid.float().mean():.3f}, d={tokens.shape[-1]})",
        flush=True,
    )
    return aligned, aligned_valid


def load_stats_block(
    cache_dir: Path, split: str, item_ids: np.ndarray, zscore: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """L1 baseline: raw per-patch stats (+ patch centres) as proxy mesh tokens.

    This is the cheap falsification test for the whole encoder investment: if the
    raw 16 statistics as tokens already reach the probe, a latent that throws
    detail away has nothing left to add.
    """
    tk = mt.load_cache(cache_dir / "mesh_tokens" / split)
    tok_ids = [str(v) for v in tk["item_ids"]]
    row = {item: i for i, item in enumerate(tok_ids)}
    order = np.array([row.get(item, -1) for item in item_ids], dtype=np.int64)
    missing = int((order < 0).sum())
    filled = np.where(order >= 0, order, 0)
    stats = np.asarray(tk["stats"], dtype=np.float32)[filled]
    centers = np.asarray(tk["centers"], dtype=np.float32)[filled]
    valid = np.asarray(tk["valid"], dtype=bool)[filled]
    valid[order < 0] = False
    feats = np.concatenate([stats, centers], axis=-1)
    feats[~valid] = 0.0
    if zscore:
        flat = feats[valid]
        mean = flat.mean(0)
        std = np.maximum(flat.std(0), 1e-6)
        feats = (feats - mean) / std
        feats[~valid] = 0.0
    print(
        f"  stats block: {len(item_ids)} rows ({missing} missing, valid patches {valid.mean():.3f}, "
        f"dim {feats.shape[-1]})",
        flush=True,
    )
    return torch.from_numpy(feats.astype(np.float16)), torch.from_numpy(valid)


class MeshDataset:
    """Probe ``Dataset`` plus per-patch mesh tokens; everything else delegates."""

    def __init__(self, ds: M.Dataset, tokens: torch.Tensor, valid: torch.Tensor):
        self._ds = ds
        self.mesh_tokens = tokens
        self.mesh_valid = valid

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __len__(self) -> int:
        return len(self._ds)


class MeshTokenProbe(M.AttentiveProbe):
    """Shipped probe with a mesh-patch block appended to the token sequence."""

    def __init__(self, cfg: M.ProbeConfig, mesh_dim: int = 256, null_embed: bool = True):
        super().__init__(cfg)
        d = cfg.d
        self.proj_mesh = nn.Linear(mesh_dim, d)
        self.mesh_type = nn.Parameter(torch.zeros(1, 1, d))
        self.register_buffer("null_vector", torch.zeros(1, 1, d), persistent=not null_embed)
        nn.init.trunc_normal_(self.proj_mesh.weight, std=0.02)
        nn.init.zeros_(self.proj_mesh.bias)
        nn.init.trunc_normal_(self.mesh_type, std=0.02)
        if null_embed:
            nn.init.trunc_normal_(self.null_vector, std=0.02)

    def tokens(self, mean, max_, grid_feats, geom, mesh=None):
        x = super().tokens(mean, max_, grid_feats, geom)
        if mesh is None:
            return x
        emb, valid = mesh
        dtype = x.dtype
        m = self.proj_mesh(emb.to(dtype)).to(dtype)
        m = m + self.mesh_type.to(dtype)
        m = m + (~valid).to(dtype).unsqueeze(-1) * self.null_vector.to(dtype)
        return torch.cat([x, m], dim=1)

    def forward(self, mean, max_, grid_feats, geom, mesh=None):
        x = self.blocks(self.tokens(mean, max_, grid_feats, geom, mesh))
        return self.head(self.norm(x[:, -1]))


def run_cv_checkpointed(ds, cfg, geom_z, out_dir: Path, k: int = 5, fold_seed: int = 0,
                        model_seed: int = 0, only_folds=None, log=print, **train_kwargs):
    """``model.run_cv`` with a per-fold OOF checkpoint and fold-level resumption.

    Mirrors the shipped recipe exactly (same ``model.folds`` split, same
    ``model.train_model`` call, same seeds); the only addition is that each
    finished fold is written to ``oof_partial.npz`` and skipped if already there,
    so stopping the run can cost at most the fold in flight.
    """
    n = len(ds)
    oof = np.zeros((n, len(M.DEFECTS)), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    partial = out_dir / "oof_partial.npz"
    if partial.exists():
        blob = np.load(partial)
        oof, done = blob["oof"], blob["done"]
        log(f"  resuming: {int(done.sum())}/{n} rows already scored")
    split = M.folds(n, k, fold_seed)
    for f in range(k) if only_folds is None else only_folds:
        val = split[f]
        if done[val].all():
            log(f"  fold {f + 1}/{k}: cached")
            continue
        train = np.setdiff1d(np.arange(n), val, assume_unique=False)
        t0 = time.time()
        model, hist = M.train_model(ds, cfg, geom_z, train, seed=model_seed + f, **train_kwargs)
        oof[val] = M.predict_probs(model, ds, geom_z, val)
        done[val] = True
        np.savez(partial, oof=oof, done=done)
        log(f"  fold {f + 1}/{k}: loss {hist[-1]:.4f} ({time.time() - t0:.0f}s) -> {partial.name}", flush=True)
    return oof, done


class JointMeshProbe(MeshTokenProbe):
    """Mesh tokens produced *inside* the probe, so the encoder sees task gradients.

    L4 of the ladder: the JEPA checkpoint is a warm start, not a frozen feature
    extractor.  Gradients reach the tokenizer + encoder through the probe loss.
    """

    def __init__(self, cfg: M.ProbeConfig, encoder: "jepa.MeshJepa", null_embed: bool = True):
        super().__init__(cfg, mesh_dim=encoder.cfg.d, null_embed=null_embed)
        self.encoder = encoder

    def encode(self, raw) -> tuple[torch.Tensor, torch.Tensor]:
        pts, stats, centers, valid, globs = raw
        valid = valid & self.encoder._rescue_empty(valid)
        h, _ = self.encoder.embed(pts, stats, centers, valid, globs)
        h = self.encoder.enc_norm(self.encoder.encoder(h, src_key_padding_mask=~valid))
        return h, valid

    def forward(self, mean, max_, grid_feats, geom, raw=None):
        mesh = None if raw is None else self.encode(raw)
        return super().forward(mean, max_, grid_feats, geom, mesh=mesh)


def make_joint_factory(state: dict, jcfg, null_embed: bool = True):
    """One fresh encoder per model (i.e. per CV fold), warm-started from ``state``.

    Without the deep copy every fold would inherit the weights the previous fold
    trained -- the encoder would have seen the next fold's validation rows and the
    OOF would be optimistic.
    """
    frozen_state = {k: v.detach().clone() for k, v in state.items()}

    def factory(cfg):
        encoder = jepa.MeshJepa(jcfg)
        encoder.load_state_dict(frozen_state)
        return JointMeshProbe(cfg, encoder=encoder, null_embed=null_embed)

    return factory


class RawBatchDataset(MeshDataset):
    """Probe rows -> raw patch batches from the memmap cache (for joint training)."""

    def __init__(self, ds: M.Dataset, batcher, order: np.ndarray, valid: torch.Tensor):
        self._ds = ds
        self.batcher = batcher
        self.order = order  # probe row -> tokeniser row (-1 where the item was skipped)
        self.mesh_tokens = None
        self.mesh_valid = valid

    def raw_batch(self, idx: np.ndarray, device) -> tuple[torch.Tensor, ...]:
        rows = self.order[idx]
        batch = self.batcher.gather(np.clip(rows, 0, None))
        valid = batch["valid"]
        missing = torch.from_numpy(rows < 0).to(device)
        if bool(missing.any()):
            valid = valid & ~missing[:, None]  # items the tokeniser skipped stay empty
        return (batch["pts"], batch["stats"], batch["centers"], valid, batch["globals"])


def tokeniser_order(dump_stem: Path, split: str, item_ids: np.ndarray, cache_dir: Path) -> np.ndarray:
    """Probe row -> tokeniser row mapping (-1 where the tokeniser skipped the item)."""
    ids_path = cache_dir / "mesh_tokens" / split / "item_ids.txt"
    tok_ids = [line.strip() for line in ids_path.read_text().splitlines()]
    row = {item: i for i, item in enumerate(tok_ids)}
    return np.array([row.get(item, -1) for item in item_ids], dtype=np.int64)


@contextlib.contextmanager
def mesh_probe(mesh_dim: int | None = None, null_embed: bool = True, factory=None):
    """Route ``train_model``/``predict_probs``/``_mean_loss`` through the mesh probe."""

    def tensors(ds, idx, geom_z, device):
        mesh = None
        if getattr(ds, "mesh_tokens", None) is not None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            mesh = (
                ds.mesh_tokens[idx].to(device, dtype, non_blocking=True),
                ds.mesh_valid[idx].to(device, non_blocking=True),
            )
        elif hasattr(ds, "raw_batch"):
            mesh = ds.raw_batch(idx, device)
        return (*_base_tensors(ds, idx, geom_z, device), mesh)

    cls = factory or functools.partial(MeshTokenProbe, mesh_dim=mesh_dim, null_embed=null_embed)
    old_tensors, old_cls = M._tensors, M.AttentiveProbe
    M._tensors = tensors
    M.AttentiveProbe = cls
    try:
        yield
    finally:
        M._tensors, M.AttentiveProbe = old_tensors, old_cls


def report(oof: np.ndarray, done: np.ndarray, y: np.ndarray, tag: str, log=print) -> dict:
    """Threshold-tuned score on the evaluated rows, plus a per-label F1 table."""
    yq = metric.derive_quality(y)
    res = metric.tune_thresholds(oof[done], y[done], yq[done], verbose=False)[1]
    per_label = " ".join(f"{k}={v:.2f}" for k, v in res["per_label"].items() if v >= 0.10)
    log(f"  {tag:28s} score={res['score']:6.3f}  artefact={res['artefact_f1_weighted']:.3f}  quality={res['quality_f1']:.3f}")
    log(f"  {' ' * 28} {per_label}")
    return {"tag": tag, **{k: float(v) for k, v in res["per_label"].items()},
            "score": float(res["score"]), "artefact": float(res["artefact_f1_weighted"]),
            "quality": float(res["quality_f1"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--data-dir", type=Path, default=TASK / "data")
    ap.add_argument("--model", default="s", help="DINOv3 cache key (shipped recipe: s)")
    ap.add_argument("--dump-stem", type=Path, default=None, help="path stem of the JEPA train dump")
    ap.add_argument("--mesh-source", choices=["dump", "stats"], default="dump",
                    help="dump = frozen JEPA tokens, stats = raw patch statistics (L1)")
    ap.add_argument("--train-encoder", action="store_true",
                    help="L4: run the encoder inside the probe with gradients (no frozen dump)")
    ap.add_argument("--ckpt", type=Path, default=None, help="JEPA checkpoint for --train-encoder")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="truncate the dataset (debug smoke tests)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", default=None, help="comma-separated fold indices (screening)")
    ap.add_argument("--epochs", type=int, default=M.TrainRecipe.epochs)
    ap.add_argument("--batch", type=int, default=M.TrainRecipe.batch)
    ap.add_argument("--lr", type=float, default=M.TrainRecipe.lr)
    ap.add_argument("--seed", type=int, default=M.TrainRecipe.seed)
    ap.add_argument("--null-embed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--baseline", action="store_true", help="also run the pristine recipe without mesh tokens")
    ap.add_argument("--reference", action="store_true", help="score the shipped OOF artifacts as reference rows")
    ap.add_argument("--out", type=Path, default=None, help="where to write OOF npy + summary json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args()

    out = args.out or (args.cache_dir / "l3_mesh_tokens" / Path(args.dump_stem).parent.name)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    only_folds = None
    if args.only_folds:
        only_folds = [int(v) for v in args.only_folds.split(",")]

    t0 = time.time()
    ds = M.load_dataset(args.cache_dir, args.data_dir, args.model, args.split, limit=args.limit)
    y = np.asarray(ds.labels)
    yq = metric.derive_quality(y)
    geom_z = ds.standardise(*M.geometry_stats(ds.geom))
    cfg = M.ProbeConfig(n_geom=ds.geom.shape[1])
    print(f"{args.split}: {len(ds)} items, {ds.geom.shape[1]} geometry features, probe d={cfg.d} depth={cfg.depth}", flush=True)

    rows = []
    common = dict(k=args.folds, fold_seed=args.seed, model_seed=args.seed, only_folds=only_folds,
                  epochs=args.epochs, batch=args.batch, lr=args.lr, log_every=args.log_every, device=device)

    if args.reference:
        for name in ("oof_probe_s.npy", "oof_fuse_s+b.npy", "oof_histgb_screen_geometry.npy"):
            path = args.cache_dir / name
            if not path.exists():
                continue
            oof = np.load(path)
            rows.append(report(oof, np.ones(len(oof), bool), y, f"reference:{name.replace('oof_', '').replace('.npy', '')}"))
        print(f"  reference rows scored in {time.time() - t0:.0f}s", flush=True)

    if args.baseline:
        t1 = time.time()
        oof, done = run_cv_checkpointed(ds, cfg, geom_z, out, **common)
        rows.append(report(oof, done, y, "baseline (shipped recipe)"))
        np.save(out / "oof_baseline.npy", oof)
        print(f"  baseline in {time.time() - t1:.0f}s", flush=True)

    tokens, valid = (
        load_mesh_block(args.dump_stem, args.split, ds.item_ids, args.cache_dir)
        if args.mesh_source == "dump"
        else load_stats_block(args.cache_dir, args.split, ds.item_ids)
    ) if not args.train_encoder else (None, None)
    t1 = time.time()
    if args.train_encoder:
        if args.ckpt is None:
            raise SystemExit("--train-encoder needs --ckpt (the JEPA checkpoint to warm-start from)")
        stem = args.dump_stem or (args.ckpt.parent / f"e_{args.split}")
        order = tokeniser_order(stem, args.split, ds.item_ids, args.cache_dir)
        batcher = jepa.MemmapBatcher(args.cache_dir / "mesh_tokens", [args.split], args.batch,
                                     torch.device(device), shuffle=False)
        valid_rows = torch.from_numpy(order >= 0)
        mesh_ds = RawBatchDataset(ds, batcher, order, valid_rows)
        encoder, jcfg = jepa.load_checkpoint(args.ckpt, device="cpu")
        factory = make_joint_factory(encoder.state_dict(), jcfg, null_embed=args.null_embed)
        print(f"  joint training: encoder d={jcfg.d} depth={jcfg.depth} from {Path(args.ckpt).name}, "
              f"gradients on, fresh encoder per fold, batch {args.batch}", flush=True)
        tag = f"joint encoder ({jcfg.d}d)"
        with mesh_probe(factory=factory):
            oof, done = run_cv_checkpointed(mesh_ds, cfg, geom_z, out, **common)
    else:
        mesh_ds = MeshDataset(ds, tokens, valid)
        tag = f"+{args.mesh_source} tokens ({int(tokens.shape[1])})"
        with mesh_probe(mesh_dim=int(tokens.shape[-1]), null_embed=args.null_embed):
            oof, done = run_cv_checkpointed(mesh_ds, cfg, geom_z, out, **common)
    rows.append(report(oof, done, y, tag))
    np.save(out / "oof_mesh.npy", oof)
    print(f"  mesh run in {time.time() - t1:.0f}s", flush=True)

    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}/summary.json (total {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
