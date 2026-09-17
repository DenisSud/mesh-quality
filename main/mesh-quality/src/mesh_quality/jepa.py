"""Mesh-patch JEPA encoder (LeJEPA objective: masked-patch prediction + SIGReg).

Trained on the tokenizer cache (``cache/mesh_tokens/<split>``); the same module
dumps frozen patch embeddings for the attentive probe (see
``knowledge/mesh-jepa-plan.md``).

Objective (per batch):

    h_target = encoder(full patches)                       # stop-grad, no EMA teacher
    h_pred   = predictor(encoder(visible patches), mask queries)
    L        = SmoothL1(h_pred, stopgrad(h_target))                 [masked positions]
             + w_stats * SmoothL1(stats_head(h_pred), stats_z)      [local measurements]
             + lam * (SIGReg(student tokens) + SIGReg(pooled))      [anti-collapse]

Masking follows the DINOv2/iBOT block recipe (ratio sampled in ``mask_lo..hi``,
applied with ``mask_probability``) but over *spatial* blocks of patch centres,
because our patches are unordered surface regions and 3D-JEPA shows region masks
beat random ones.

The 16 hand-computed patch statistics go into the tokenizer *and* are kept as raw
tokens for the probe: the JEPA latent is free to discard unpredictable detail,
the probe must not have to rely on it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mesh_tokens as mt

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass
class JepaConfig:
    # tokenizer / encoder
    d: int = 256
    depth: int = 6
    heads: int = 4
    point_hidden: int = 64
    dropout: float = 0.0
    # predictor
    pred_d: int = 192
    pred_depth: int = 4
    pred_heads: int = 4
    # objective
    mask_lo: float = 0.15
    mask_hi: float = 0.40
    mask_blocks: tuple[int, int] = (1, 3)
    mask_probability: float = 0.75   # 25% of steps are view-only (no mask)
    pred_beta: float = 1.0
    stats_weight: float = 0.10
    sigreg_weight: float = 0.05
    sigreg_slices: int = 128
    sigreg_knots: int = 17
    sigreg_tmax: float = 3.0
    sigreg_max_tokens: int = 8192
    # augmentation (applied to cached tensors, per step)
    point_dropout: float = 0.20
    point_jitter: float = 0.02       # relative to the patch radius channel
    patch_dropout: float = 0.10      # student view only
    # optimisation
    lr: float = 5e-4
    wd: float = 0.05
    warmup: float = 0.05
    clip: float = 1.0
    batch: int = 256
    epochs: int = 500
    amp: bool = True
    workers: int = 4


# --------------------------------------------------------------------------- #
# SIGReg (Epps-Pulley characteristic-function statistic over random slices)
# --------------------------------------------------------------------------- #


def sigreg(
    z: torch.Tensor, slices: int = 128, knots: int = 17, t_max: float = 3.0,
    max_tokens: int = 8192, chunk: int = 4096, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Push every 1-D projection of ``z`` towards N(0, 1); collapse and blow-up
    are both penalised, so no EMA teacher or stop-gradient heuristics are needed.

    The statistic is a mean over tokens, so it is estimated on a random token
    subsample (``max_tokens``) and accumulated in chunks: the naive
    ``(N, slices, knots)`` intermediate costs >1 GB of activations at batch 256.
    """
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] < 2:
        return z.new_zeros(())
    if z.shape[0] > max_tokens:
        idx = torch.randperm(z.shape[0], generator=generator, device=z.device)[:max_tokens]
        z = z[idx]
    a = torch.randn(slices, z.shape[1], device=z.device, dtype=z.dtype, generator=generator)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = torch.linspace(-t_max, t_max, knots, device=z.device, dtype=z.dtype)
    cos_sum = torch.zeros(slices, knots, device=z.device, dtype=z.dtype)
    sin_sum = torch.zeros_like(cos_sum)
    for start in range(0, z.shape[0], chunk):
        tp = (z[start:start + chunk] @ a.t()).unsqueeze(-1) * t          # (c, S, K)
        cos_sum = cos_sum + tp.cos().sum(0)
        sin_sum = sin_sum + tp.sin().sum(0)
    cos = cos_sum / z.shape[0]
    sin = sin_sum / z.shape[0]
    err = (cos - torch.exp(-0.5 * t**2)[None, :]) ** 2 + sin**2
    w = torch.full((knots,), 0.5, device=z.device, dtype=z.dtype)
    w[0] = w[-1] = 0.25 if knots > 1 else w[0]
    w = w * (t[1] - t[0] if knots > 1 else 1.0)
    return (err * w[None, :]).sum(-1).mean()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class PatchTokenizer(nn.Module):
    """Per-patch point features + measurements + centre -> one token.

    Pools *before* expanding to ``d``: the pointwise MLP sees 32 points per patch,
    so expanding to d first made the tokenizer ~30x the encoder's FLOPs and its
    activations dominated memory.
    """

    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(len(mt.POINT_NAMES), cfg.point_hidden), nn.GELU()
        )
        self.merge = nn.Linear(cfg.point_hidden, cfg.d)
        self.stats = nn.Linear(len(mt.STAT_NAMES), cfg.d)
        self.center = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, cfg.d))
        self.globals = nn.Linear(len(mt.GLOBAL_NAMES), cfg.d)   # mesh-level context (scale, density, flags)
        self.norm = nn.LayerNorm(cfg.d)

    def forward(
        self, pts: torch.Tensor, stats: torch.Tensor, centers: torch.Tensor, globals_z: torch.Tensor
    ) -> torch.Tensor:
        h = self.merge(self.point(pts).amax(dim=2))          # (B, K, point_hidden) -> (B, K, d)
        h = h + self.stats(stats) + self.center(centers) + self.globals(globals_z)[:, None, :]
        return self.norm(h)


class MaskToken(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.base, std=0.02)

    def forward(self, centers: torch.Tensor, pe: nn.Module) -> torch.Tensor:
        return self.base.expand(centers.shape[0], centers.shape[1], -1) + pe(centers)


class MeshJepa(nn.Module):
    def __init__(self, cfg: JepaConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = PatchTokenizer(cfg)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.d, cfg.heads, 4 * cfg.d, dropout=cfg.dropout, activation="gelu",
                norm_first=True, batch_first=True,
            ),
            cfg.depth, enable_nested_tensor=False,
        )
        self.enc_norm = nn.LayerNorm(cfg.d)
        self.mask_token = MaskToken(cfg.d)
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.pred_d, cfg.pred_heads, 4 * cfg.pred_d, dropout=cfg.dropout, activation="gelu",
                norm_first=True, batch_first=True,
            ),
            cfg.pred_depth, enable_nested_tensor=False,
        )
        self.pred_in = nn.Linear(cfg.d, cfg.pred_d)
        self.pred_out = nn.Linear(cfg.pred_d, cfg.d)
        self.stats_head = nn.Sequential(nn.Linear(cfg.d, 128), nn.GELU(), nn.Linear(128, len(mt.STAT_NAMES)))
        for name, dim in (
            ("point_mean", len(mt.POINT_NAMES)), ("point_std", len(mt.POINT_NAMES)),
            ("stat_mean", len(mt.STAT_NAMES)), ("stat_std", len(mt.STAT_NAMES)),
            ("global_mean", len(mt.GLOBAL_NAMES)), ("global_std", len(mt.GLOBAL_NAMES)),
        ):
            self.register_buffer(name, torch.zeros(dim) if name.endswith("mean") else torch.ones(dim))

    # -- tokenizer --------------------------------------------------------- #

    def set_feature_norm(self, norm: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for key, value in norm.items():
                getattr(self, key).copy_(value.float())

    def embed(
        self, pts: torch.Tensor, stats: torch.Tensor, centers: torch.Tensor,
        valid: torch.Tensor | None = None, globals_: torch.Tensor | None = None, augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizer with everything in bfloat16 where the kernels allow it.

        Matmuls under autocast are bf16 anyway; keeping the elementwise
        normalisation/augmentation in fp32 doubled the memory traffic of the
        (B, K, k, F) tensors for nothing.
        """
        dtype = torch.bfloat16 if (self.cfg.amp and pts.is_cuda) else torch.float32
        pts = pts.to(dtype)
        stats = stats.to(dtype)
        p_mean, p_std = self.point_mean.to(dtype), self.point_std.to(dtype)
        s_mean, s_std = self.stat_mean.to(dtype), self.stat_std.to(dtype)
        globals_z = (
            (globals_.to(dtype) - self.global_mean.to(dtype)) / self.global_std.to(dtype)
            if globals_ is not None
            else torch.zeros(pts.shape[0], self.global_mean.shape[0], device=pts.device, dtype=dtype)
        )
        if augment:
            cfg = self.cfg
            if cfg.point_dropout > 0:
                keep = torch.rand_like(pts[..., 0]) > cfg.point_dropout
                pts = pts * keep.unsqueeze(-1)
            if cfg.point_jitter > 0:
                pts = pts + torch.randn_like(pts) * cfg.point_jitter * pts[..., -1:].clamp_min(0.0)
        pts = (pts - p_mean) / p_std
        stats_z = (stats - s_mean) / s_std
        if valid is not None:
            pts = pts * valid[:, :, None, None]
            stats_z = stats_z * valid[:, :, None]
        return self.tokenizer(pts, stats_z, centers.to(dtype), globals_z), stats_z.float()

    # -- forward ----------------------------------------------------------- #

    def forward(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor | float | None]:
        cfg = self.cfg
        pts, stats, centers, valid = batch["pts"], batch["stats"], batch["centers"], batch["valid"]
        globals_ = batch.get("globals")
        valid = valid & self._rescue_empty(valid)
        mask = self.sample_mask(centers, valid, generator)
        student_valid = valid & ~self._patch_dropout(valid, generator)

        h_full, stats_z = self.embed(pts, stats, centers, valid, globals_)
        with torch.no_grad():
            target = self.enc_norm(self.encoder(h_full, src_key_padding_mask=~valid))
        stats_z = stats_z.detach()

        h_student, _ = self.embed(pts, stats, centers, student_valid, globals_, augment=True)
        pad = ~student_valid
        h_enc = self.enc_norm(self.encoder(h_student, src_key_padding_mask=pad))

        context = ~mask & student_valid                      # what the predictor may see
        return self._predict(cfg, h_enc, target, stats_z, centers, context, mask & valid)

    def _predict(
        self, cfg: JepaConfig, h_enc: torch.Tensor, target: torch.Tensor, stats_z: torch.Tensor,
        centers: torch.Tensor, context: torch.Tensor, masked: torch.Tensor,
    ) -> dict[str, Any]:
        # fixed shapes (no .item() syncs, CUDA-graph friendly): the predictor sees
        # every token with a padding mask plus a fixed number of mask queries
        b, k, d = h_enc.shape
        max_masked = int(math.ceil(cfg.mask_hi * k)) + 2
        pooled = self._pool(h_enc, context)
        sig_kw = dict(slices=cfg.sigreg_slices, knots=cfg.sigreg_knots, t_max=cfg.sigreg_tmax,
                      max_tokens=cfg.sigreg_max_tokens, generator=None)
        if not bool(masked.any()):
            loss_sig = sigreg(h_enc[context], **sig_kw) + sigreg(pooled, **sig_kw)
            zero = loss_sig.detach() * 0
            return {"loss": cfg.sigreg_weight * loss_sig, "pred": zero, "stats": zero,
                    "sigreg": loss_sig.detach(), "masked_frac": zero, "pooled": pooled.detach()}

        pos = torch.argsort(~masked, dim=1, stable=True)[:, :max_masked]
        q_pad = torch.arange(max_masked, device=h_enc.device)[None, :] >= masked.sum(dim=1, keepdim=True)
        query_centers = torch.gather(centers, 1, pos[..., None].expand(-1, -1, 3)).float()
        queries = self.pred_in(self.mask_token(query_centers, self.tokenizer.center))

        seq = torch.cat([self.pred_in(h_enc), queries], dim=1)
        out = self.predictor(seq, src_key_padding_mask=torch.cat([~context, q_pad], dim=1))[:, k:]
        pred = self.pred_out(out)

        okay = ~q_pad
        tgt = torch.gather(target, 1, pos[..., None].expand(-1, -1, d))
        loss_pred = F.smooth_l1_loss(pred[okay], tgt[okay], beta=cfg.pred_beta)
        stat_tgt = torch.gather(stats_z, 1, pos[..., None].expand(-1, -1, stats_z.shape[-1]))
        loss_stats = F.smooth_l1_loss(self.stats_head(pred[okay]), stat_tgt[okay], beta=cfg.pred_beta)

        student_all = torch.cat([h_enc[context], pred[okay]], dim=0)
        loss_sig = sigreg(student_all, **sig_kw) + sigreg(pooled, **sig_kw)
        total = loss_pred + cfg.stats_weight * loss_stats + cfg.sigreg_weight * loss_sig
        return {"loss": total, "pred": loss_pred.detach(), "stats": loss_stats.detach(),
                "sigreg": loss_sig.detach(), "masked_frac": masked.float().mean(),
                "pooled": pooled.detach()}

    @staticmethod
    def _pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vm = mask[:, :, None].float()
        return (h * vm).sum(1) / vm.sum(1).clamp_min(1)

    # -- masking helpers --------------------------------------------------- #

    @staticmethod
    def _rescue_empty(valid: torch.Tensor) -> torch.Tensor:
        """Items with no valid patch would produce NaN attention: keep one slot."""
        empty = valid.sum(dim=1) == 0
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty, 0] = True
        return valid

    def _patch_dropout(self, valid: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if self.cfg.patch_dropout <= 0:
            return torch.zeros_like(valid)
        drop = torch.rand(valid.shape, generator=generator, device=valid.device) < self.cfg.patch_dropout
        return drop & valid

    def sample_mask(
        self, centers: torch.Tensor, valid: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Spatial block mask over valid patches; ratio in [mask_lo, mask_hi].

        Fully batched: the per-item Python version cost ~30 ms/step (kernel-launch
        bound) versus a few ms here, and it must leave >=3 context patches.
        """
        cfg = self.cfg
        b, k, _ = centers.shape
        dev = centers.device
        lo, hi = cfg.mask_blocks
        nb_max = min(int(hi), k)
        rnd = lambda *shape: torch.rand(*shape, generator=generator, device=dev)  # noqa: E731

        n_valid = valid.sum(dim=1)
        active = (rnd(b) < cfg.mask_probability) & (n_valid > 3)
        ratios = cfg.mask_lo + (cfg.mask_hi - cfg.mask_lo) * rnd(b)
        nb = torch.randint(lo, hi + 1, (b,), generator=generator, device=dev).clamp(max=nb_max)

        seeds = torch.rand(b, k, generator=generator, device=dev).masked_fill(~valid, -1.0).topk(nb_max, dim=1).indices
        seed_ok = torch.arange(nb_max, device=dev)[None, :] < nb[:, None]                      # (B, nb)
        seed_c = torch.gather(centers, 1, seeds[..., None].expand(-1, -1, 3))                  # (B, nb, 3)
        d2 = torch.cdist(centers, seed_c) ** 2                                                 # (B, K, nb)
        d2 = d2.masked_fill(~seed_ok[:, None, :], float("inf"))
        n_mask = (ratios * n_valid / nb.clamp_min(1)).round().clamp_min(1).long()               # (B,)
        m = min(k - 1, int(math.ceil(cfg.mask_hi * k)) + 2)
        take = d2.topk(m, dim=1, largest=False).indices                                        # (B, m, nb)
        ok = seed_ok[:, None, :] & (torch.arange(m, device=dev)[None, :, None] < n_mask[:, None, None])

        flat_idx = take.permute(0, 2, 1).reshape(b, -1)
        flat_ok = ok.permute(0, 2, 1).reshape(b, -1)
        flat_idx = torch.where(flat_ok, flat_idx, torch.zeros_like(flat_idx))
        mask = torch.zeros(b, k, dtype=torch.bool, device=dev)
        mask.scatter_(1, flat_idx, flat_ok)

        # clamp: keep >=3 context patches per item (>=20% of the valid surface)
        budget = (n_valid - torch.maximum(torch.full_like(n_valid, 3), (0.2 * n_valid).long())).clamp_min(1)
        prio = torch.where(mask, rnd(b, k), torch.full((b, k), 2.0, device=dev))                # unmasked last
        keep = prio.argsort(dim=1).argsort(dim=1) < budget[:, None]
        mask = mask & keep & valid & active[:, None]
        return mask


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


class MeshPatchDataset(torch.utils.data.Dataset):
    """Row-indexed view over the flat tokenizer cache (memmap, no per-item files)."""

    def __init__(self, cache_dir: str | Path, split: str):
        cache = mt.load_cache(Path(cache_dir) / split)
        self.pts = cache["pts"]
        self.stats = cache["stats"]
        self.centers = cache["centers"]
        self.valid = cache["valid"]
        self.globals = cache["globals"]
        self.item_ids = list(cache["item_ids"])
        self.split = split
        self.n_patch = int(self.stats.shape[1])
        self.n_stats = int(self.stats.shape[2])

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "pts": torch.tensor(np.asarray(self.pts[i], dtype=np.float32), dtype=torch.bfloat16),
            "stats": torch.tensor(np.asarray(self.stats[i], dtype=np.float32), dtype=torch.bfloat16),
            "centers": torch.tensor(np.asarray(self.centers[i], dtype=np.float32)),
            "valid": torch.tensor(np.asarray(self.valid[i], dtype=np.bool_)),
            "globals": torch.tensor(np.asarray(self.globals[i], dtype=np.float32)),
            "index": torch.tensor(i, dtype=torch.long),
        }


class MultiMeshPatchDataset(torch.utils.data.Dataset):
    """Concatenation of several split caches (pretraining uses train+test: the
    objective is unsupervised, so the unlabeled test meshes are extra data)."""

    def __init__(self, cache_dir: str | Path, splits: Sequence[str]):
        self.parts = [MeshPatchDataset(cache_dir, s) for s in splits]
        self.cum = np.cumsum([len(p) for p in self.parts])
        self.item_ids = [i for part in self.parts for i in part.item_ids]
        self.n_patch = self.parts[0].n_patch
        self.n_stats = self.parts[0].n_stats
        self.splits = list(splits)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k = int(np.searchsorted(self.cum, i, side="right"))
        local = i - (int(self.cum[k - 1]) if k else 0)
        item = self.parts[k][local]
        item["index"] = torch.tensor(i, dtype=torch.long)   # global row id
        return item


def compute_feature_stats(dataset, n: int = 512, seed: int = 0) -> dict[str, torch.Tensor]:
    """Per-feature mean/std over valid patches of a random item sample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    items = [dataset[int(i)] for i in idx]
    pts = torch.stack([it["pts"] for it in items])              # (n, K, k, F)
    stats = torch.stack([it["stats"] for it in items])
    valid = torch.stack([it["valid"] for it in items])
    glob = torch.stack([it["globals"] for it in items])
    keep = valid.reshape(-1)
    pts = pts.reshape(-1, *pts.shape[2:])[keep].float()
    stats = stats.reshape(-1, stats.shape[2])[keep].float()
    return {
        "point_mean": pts.reshape(-1, pts.shape[-1]).mean(0),
        "point_std": pts.reshape(-1, pts.shape[-1]).std(0).clamp_min(1e-3),
        "stat_mean": stats.mean(0),
        "stat_std": stats.std(0).clamp_min(1e-3),
        "global_mean": glob.mean(0),
        "global_std": glob.std(0).clamp_min(1e-3),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(
    cache_dir: str | Path,
    out_dir: str | Path,
    cfg: JepaConfig,
    split: str = "train",
    labels: np.ndarray | None = None,
    label_rows: int | None = None,
    log_every: int = 25,
    eval_every: int = 50,
    save_every: int = 100,
    device: str | None = None,
    seed: int = 0,
    max_minutes: float | None = None,
) -> Path:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")   # TF32 for the fp32 parts (losses)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(cache_dir, split)
    splits = ["train", "test"] if split == "both" else [split]
    batcher = MemmapBatcher(cache_dir, splits, cfg.batch, dev, shuffle=True, seed=seed)
    model = MeshJepa(cfg).to(dev)
    model.set_feature_norm(compute_feature_stats(dataset))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps_per_epoch = len(batcher)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return cfg.lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * cfg.lr * (1 + math.cos(math.pi * progress)) + 1e-6

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    log: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = {"loss": 0.0, "pred": 0.0, "stats": 0.0, "sigreg": 0.0, "masked_frac": 0.0}
        n_batches = 0
        for batch in batcher:
            batch = _move(batch, dev)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
                result = model({k: v for k, v in batch.items() if k != "index"}, generator=gen)
            opt.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            step += 1
            n_batches += 1
            for key in running:
                value = result[key]
                running[key] += float(value.detach()) if torch.is_tensor(value) else float(value)
        means = {k: v / max(n_batches, 1) for k, v in running.items()}
        means.update({"epoch": epoch, "lr": lr_at(step), "min": (time.time() - start) / 60})
        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:4d} loss {means['loss']:.4f} pred {means['pred']:.4f} "
                  f"stats {means['stats']:.4f} sigreg {means['sigreg']:.4f} "
                  f"mask {means['masked_frac']:.2f} lr {means['lr']:.1e} {means['min']:.1f}min", flush=True)
        if eval_every and (epoch % eval_every == 0 or epoch == cfg.epochs - 1):
            metrics = monitor(model, dataset, dev, labels=labels, label_rows=label_rows)
            means.update(metrics)
            print(f"  monitor {json.dumps({k: round(v, 4) for k, v in metrics.items()})}", flush=True)
        log.append(means)
        if save_every and (epoch % save_every == 0 or epoch == cfg.epochs - 1):
            save_checkpoint(model, cfg, out / "ckpt.pt", epoch, log)
        if max_minutes and (time.time() - start) / 60 > max_minutes:
            print(f"time budget reached at epoch {epoch}", flush=True)
            break
    save_checkpoint(model, cfg, out / "ckpt.pt", epoch, log)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    return out / "ckpt.pt"


def save_checkpoint(model: MeshJepa, cfg: JepaConfig, path: Path, epoch: int, log: list[dict]) -> None:
    norm = {k: getattr(model, k).detach().cpu() for k in
            ("point_mean", "point_std", "stat_mean", "stat_std", "global_mean", "global_std")}
    torch.save(
        {"config": asdict(cfg), "state": {k: v.cpu() for k, v in model.state_dict().items()},
         "feature_norm": norm, "epoch": epoch, "log": log[-20:]},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[MeshJepa, JepaConfig]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = JepaConfig(**blob["config"])
    model = MeshJepa(cfg)
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    return model, cfg


# --------------------------------------------------------------------------- #
# frozen embeddings / monitoring
# --------------------------------------------------------------------------- #


class MemmapBatcher:
    """Batches straight from the memmaps: one fancy-index per array per batch.

    The per-item DataLoader path cost more CPU than the GPU step (GPU sat at 4%),
    so gather a whole batch at once instead of per-item python plus IPC.
    """

    KEYS = ("pts", "stats", "centers", "valid", "globals")

    def __init__(self, cache_dir: str | Path, splits: Sequence[str], batch: int, device: torch.device,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True):
        parts = [mt.load_cache(Path(cache_dir) / s) for s in splits]
        self.arrays = {key: [p[key] for p in parts] for key in self.KEYS}
        sizes = [len(p["voxel"]) for p in parts]
        self.cum = np.cumsum(sizes)
        self.part_of = np.repeat(np.arange(len(parts)), sizes)
        self.offset = np.concatenate([[0], self.cum[:-1]]).astype(np.int64)
        self.n = int(self.cum[-1])
        self.batch = batch
        self.device = device
        self.shuffle = shuffle
        self.seed = seed
        self.n_batches = self.n // batch if drop_last else math.ceil(self.n / batch)
        self._epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def gather(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        """Fetch arbitrary rows (probe order -> tokeniser order mapping lives outside)."""
        return self._gather(rows)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for start in range(0, self.n - self.batch + 1, self.batch):
            yield self._gather(order[start:start + self.batch])

    def _gather(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        parts = self.part_of[rows]
        local = rows - self.offset[parts]
        out: dict[str, torch.Tensor] = {}
        for key, split_arrays in self.arrays.items():
            merged = np.empty((len(rows), *split_arrays[0].shape[1:]), dtype=split_arrays[0].dtype)
            for p in np.unique(parts):
                sel = np.flatnonzero(parts == p)
                merged[sel] = split_arrays[int(p)][local[sel]]
            tensor = torch.from_numpy(merged)
            if key == "valid":
                tensor = tensor.to(torch.bool)
            elif key in ("centers", "globals"):
                tensor = tensor.float()
            out[key] = tensor.to(self.device, non_blocking=True)
        return out


def make_dataset(cache_dir: str | Path, split: str):
    """``split`` is ``train``/``test`` or ``both`` (unsupervised pretraining)."""
    if split == "both":
        return MultiMeshPatchDataset(cache_dir, ["train", "test"])
    return MeshPatchDataset(cache_dir, split)


@torch.no_grad()
def extract_embeddings(
    model: MeshJepa, dataset, device: torch.device, batch: int = 256, pool_only: bool = False,
) -> dict[str, np.ndarray]:
    """Frozen encoder pass: patch tokens (fp16) + pooled embedding + validity."""
    n, k = len(dataset), int(dataset.n_patch)
    d = model.cfg.d
    pooled = np.zeros((n, d), np.float32)
    valid_out = np.zeros((n, k), bool)
    tokens = None if pool_only else np.zeros((n, k, d), np.float16)
    model.eval()
    for start in range(0, n, batch):
        items = [dataset[i] for i in range(start, min(start + batch, n))]
        pts = torch.stack([it["pts"] for it in items]).to(device)
        stats = torch.stack([it["stats"] for it in items]).to(device)
        centers = torch.stack([it["centers"] for it in items]).to(device)
        valid = torch.stack([it["valid"] for it in items]).to(device)
        valid = valid & MeshJepa._rescue_empty(valid)
        globs = torch.stack([it["globals"] for it in items]).to(device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16,
                                  enabled=model.cfg.amp and device.type == "cuda")
        with autocast:
            h, _ = model.embed(pts, stats, centers, valid, globs)
            h = model.enc_norm(model.encoder(h, src_key_padding_mask=~valid))
        h = h.float()
        vm = valid[:, :, None].float()
        pooled[start:start + len(items)] = ((h * vm).sum(1) / vm.sum(1).clamp_min(1)).cpu().numpy()
        valid_out[start:start + len(items)] = valid.cpu().numpy()
        if tokens is not None:
            tokens[start:start + len(items)] = h.half().cpu().numpy()
    return {"pooled": pooled, "valid": valid_out, **({"tokens": tokens} if tokens is not None else {})}


def monitor(
    model: MeshJepa, dataset, device: torch.device, labels: np.ndarray | None = None,
    label_rows: int | None = None,
) -> dict[str, float]:
    """Embedding geometry (collapse detectors) + optional pooled logistic probe."""
    emb = extract_embeddings(model, dataset, device, pool_only=True)
    pooled = np.asarray(emb["pooled"], np.float32)
    sv = np.linalg.svd(pooled[: min(len(pooled), 2048)], compute_uv=False)
    p = sv / max(sv.sum(), 1e-12)
    out = {
        "emb_std": float(pooled.std(0).mean()),
        "emb_norm": float(np.linalg.norm(pooled, axis=1).mean()),
        "eff_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
    }
    rows = len(pooled) if label_rows is None else label_rows
    if labels is not None and len(labels) == rows and rows > 0:
        out.update(probe_metrics(pooled[:rows], labels))
    return out


def probe_metrics(pooled: np.ndarray, labels: np.ndarray, folds: int = 1, seed: int = 0,
                  rows: int = 6000) -> dict[str, float]:
    """Cheap pooled-embedding logistic probe used as a *trend* signal.

    One stratified split on a row subsample by default: the 5-fold version costs
    50 CPU fits per evaluation, which stalled the GPU for minutes every time it
    ran. The real 5-fold probe belongs to the deliberate L3 evaluation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from . import metric

    y = np.asarray(labels)
    if rows and len(pooled) > rows:
        idx, _ = train_test_split(np.arange(len(pooled)), train_size=rows, stratify=y[:, 0], random_state=seed)
        pooled, y = pooled[idx], y[idx]
    oof = np.zeros_like(y, dtype=np.float32)
    if folds <= 1:
        train_idx, test_idx = train_test_split(np.arange(len(pooled)), test_size=0.25,
                                               stratify=y[:, 0], random_state=seed)
        splits = [(train_idx, test_idx)]
    else:
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(pooled, y[:, 0]))
    for train_idx, test_idx in splits:
        for j in range(y.shape[1]):
            if y[:, j].min() == y[:, j].max():
                oof[test_idx, j] = float(y[:, j].mean())
                continue
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(pooled[train_idx], y[train_idx, j])
            oof[test_idx, j] = clf.predict_proba(pooled[test_idx])[:, 1]
    labeled = np.zeros(len(oof), bool)
    for _, test_idx in splits:
        labeled[test_idx] = True
    pred = (oof >= 0.5).astype(np.int8)
    res = metric.score_predictions(y[labeled], metric.derive_quality(y[labeled]),
                                   pred[labeled], metric.derive_quality(pred[labeled]))
    return {"probe_score": float(res["score"]), "probe_quality": float(res["quality_f1"]),
            "probe_noisy": float(res["per_label"]["noisy"]), "probe_lowpoly": float(res["per_label"]["lowpoly"])}


def dump_embeddings(
    checkpoint: str | Path, cache_dir: str | Path, split: str, out_stem: str | Path,
    device: str = "cuda", pool_only: bool = False,
) -> Path:
    """Write frozen patch tokens (fp16) + pooled embedding + validity for the probe."""
    model, _ = load_checkpoint(checkpoint, device=device)
    dataset = make_dataset(cache_dir, split)
    emb = extract_embeddings(model, dataset, torch.device(device), pool_only=pool_only)
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(stem.with_name(stem.name + "_pooled.npy"), emb["pooled"])
    np.save(stem.with_name(stem.name + "_valid.npy"), emb["valid"])
    if "tokens" in emb:
        np.save(stem.with_name(stem.name + "_tokens.npy"), emb["tokens"])
    return stem
