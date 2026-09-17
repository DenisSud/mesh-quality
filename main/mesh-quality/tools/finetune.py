"""End-to-end DINOv3 fine-tune under the shipped probe head (the one untried lever).

The shipped recipe keeps DINOv3 frozen and trains a 4-layer transformer on cached
tokens.  This tool unfreezes the backbone: PNG collage -> backbone (with grads)
-> the SAME token layout [6 mean | 6 max | 6x64 grid | GEOM | query] -> the SAME
``AttentiveProbe`` -> 10 sigmoids, with the SAME BCE pos_weight loss.  The only
new things are the augmentation (per-tile hflip + brightness/contrast jitter,
uint8 domain on GPU) and two learning rates (backbone ~2e-5, head ~1e-3).

    fold-0 validation run:
        devenv shell -- uv run python main/mesh-quality/tools/finetune.py \
            --out-dir main/mesh-quality/cache/ft_fold0 --val-fold 0
    refit on all rows + test predict (+hflip TTA):
        ... --refit --out-dir main/mesh-quality/cache/ft_full
    smoke (CPU, ~2 min):
        ... --limit 16 --epochs 1 --batch 2 --device cpu --val-fold 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import images, metric, model as M  # noqa: E402


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class FineTuneNet(nn.Module):
    """Trainable DINOv3 + the shipped probe; one forward = one item's 6 tiles."""

    def __init__(self, cfg: M.ProbeConfig, model_key: str = "s", grad_ckpt: bool = False):
        super().__init__()
        import timm

        self.backbone = timm.create_model(images.MODELS[model_key], pretrained=True, num_classes=0)
        if grad_ckpt and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing()
        self.probe = M.AttentiveProbe(cfg)
        self.model_key = model_key
        self.n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.n_patch = images.GRID * images.GRID
        self.dim = int(self.backbone.embed_dim)
        if self.dim != cfg.dim:
            raise ValueError(f"backbone dim {self.dim} != cfg.dim {cfg.dim}")

    def tile_features(self, tiles_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[b, 6, 512, 512, 3] uint8 (on device) -> (mean, max, pooled grid)."""
        b = tiles_u8.shape[0]
        x = tiles_u8.reshape(b * images.N_TILES, *tiles_u8.shape[-3:])
        feats = self.backbone.forward_features(_normalise_tensor(x))
        patches = feats[:, self.n_prefix : self.n_prefix + self.n_patch].float()
        patches = patches.reshape(b * images.N_TILES, images.GRID, images.GRID, self.dim)
        mean, max_, grid = images._tile_summaries(patches)  # per-tile summaries
        mean = mean.reshape(b, images.N_TILES, self.dim)
        max_ = max_.reshape(b, images.N_TILES, self.dim)
        grid = grid.reshape(b, images.N_TILES, images.POOLED_GRID, images.POOLED_GRID, self.dim)
        return mean, max_, grid

    def forward(self, tiles_u8: torch.Tensor, geom_z: torch.Tensor) -> torch.Tensor:
        mean, max_, grid = self.tile_features(tiles_u8)
        return self.probe(mean.float(), max_.float(), grid.float(), geom_z)


def augment(x: torch.Tensor, p_flip: float = 0.5, jitter: float = 0.1) -> torch.Tensor:
    """uint8 augmentation on device: per-tile hflip + per-item brightness/contrast.

    No rotations (canonical views), no crops (frame occupancy is the scale cue).
    """
    b, t = x.shape[:2]
    flip = torch.rand(b, t, 1, 1, 1, device=x.device) < p_flip
    x = torch.where(flip, x.flip(-1), x)
    a = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
    c = (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * (255.0 * jitter * 0.4)
    return (x.float() * a + c).clamp_(0, 255).to(torch.uint8)


def _normalise_tensor(x: torch.Tensor) -> torch.Tensor:
    """``[B, 512, 512, 3]`` uint8 on device -> normalised ``[B, 3, 512, 512]``.

    Same math as ``images._normalise`` but without the numpy round-trip.
    """
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(images.IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(images.IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def batch_loader(paths: list[Path], order: np.ndarray, batch: int, workers: int = 6):
    """Yield (idx, tiles_u8[b,6,512,512,3]) for the given row order, prefetching."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    window = max(batch * 4, 24)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        it = iter(order)
        def submit(i):
            pending.append((i, pool.submit(images.split_tiles, paths[i])))
        for _ in range(min(window, len(order))):
            submit(next(it))
        buf: list[tuple[int, np.ndarray]] = []
        while pending:
            i, fut = pending.popleft()
            buf.append((i, fut.result()))
            nxt = next(it, None)
            if nxt is not None:
                submit(nxt)
            if len(buf) == batch or (not pending and buf):
                ids = np.array([k for k, _ in buf])
                yield ids, torch.from_numpy(np.stack([t for _, t in buf]))
                buf = []


# --------------------------------------------------------------------------- #
# train / predict
# --------------------------------------------------------------------------- #

def train(model: FineTuneNet, ds, geom_z, paths, train_idx, device, *, epochs, batch,
          lr_backbone, lr_head, wd, seed, val_idx=None, log_every=1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = torch.from_numpy(np.asarray(ds.labels, dtype=np.float32)).to(device)
    pos = y[train_idx].sum(0)
    neg = len(train_idx) - pos
    pos_weight = torch.clamp((neg / pos.clamp(min=1.0)) ** 0.5, 1.0, 20.0).to(device)

    groups = [
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.probe.parameters(), "lr": lr_head},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    steps = int(np.ceil(len(train_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_backbone, lr_head], total_steps=epochs * steps,
        pct_start=0.1, div_factor=10.0, final_div_factor=100.0,
    )
    paths = [paths[i] for i in range(len(paths))]
    history, val_probs = [], None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total, nb = 0.0, 0
        for ids, tiles in batch_loader(paths, perm, batch):
            tiles = tiles.to(device, non_blocking=True)
            g = torch.from_numpy(geom_z[ids]).to(device)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(augment(tiles), g)
                loss = F.binary_cross_entropy_with_logits(logits, y[ids], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(ids)
            nb += 1
        history.append(total / len(perm))
        msg = f"    epoch {epoch + 1}/{epochs} loss {history[-1]:.4f}"
        if val_idx is not None:
            val_probs = predict(model, ds, geom_z, paths, val_idx, device)
            p = torch.from_numpy(np.clip(val_probs, 1e-6, 1 - 1e-6)).float()
            logits_val = torch.log(p / (1 - p))
            vl = F.binary_cross_entropy_with_logits(
                logits_val, torch.from_numpy(np.asarray(ds.labels)[val_idx]).float(),
                pos_weight=pos_weight.cpu(),
            ).item()
            msg += f"  val_bce {vl:.4f}"
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"{msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return model, history, val_probs


@torch.inference_mode()
def predict(model: FineTuneNet, ds, geom_z, paths, idx, device, batch: int = 16, tta_flip: bool = False):
    """Deterministic probabilities for rows ``idx`` (no augmentation)."""
    model.eval()
    order = np.asarray(idx)
    inv = np.full(len(paths), -1, dtype=np.int64)
    inv[order] = np.arange(len(order))
    out = np.zeros((len(order), len(metric.DEFECTS)), dtype=np.float32)
    for ids, tiles in batch_loader(paths, order, batch):
        tiles = tiles.to(device, non_blocking=True)
        g = torch.from_numpy(geom_z[ids]).to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.startswith("cuda")):
            p = torch.sigmoid(model(tiles, g)).float()
            if tta_flip:
                p = 0.5 * p + 0.5 * torch.sigmoid(model(tiles.flip(-1), g)).float()
        out[inv[ids]] = p.cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--data-dir", type=Path, default=TASK / "data")
    ap.add_argument("--model", default="s")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-fold", type=int, default=None, help="train on the other folds, score this one")
    ap.add_argument("--refit", action="store_true", help="train on ALL rows (after the fold gate passed)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6, help="items per step (x6 tiles)")
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", type=int, default=1, help="hflip TTA at test predict (0/1)")
    ap.add_argument("--predict-only", action="store_true",
                    help="load the saved ft.pt and predict test (no training)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.predict_only:
        blob = torch.load(out / "ft.pt", map_location="cpu", weights_only=False)
        cfg = M.ProbeConfig(**blob["cfg"])
        model = FineTuneNet(cfg, blob["model_key"]).to(device)
        model.backbone.load_state_dict(blob["backbone"])
        model.probe.load_state_dict(blob["probe"])
        model.eval()
        median, scale = np.asarray(blob["geom_median"]), np.asarray(blob["geom_scale"])
        ds = M.load_dataset(args.cache_dir, args.data_dir, blob["model_key"], "train", limit=args.limit)
        _predict_test(model, ds, median, scale, out, device, args)
        return 0

    ds = M.load_dataset(args.cache_dir, args.data_dir, args.model, "train", limit=args.limit)
    y = np.asarray(ds.labels)
    yq = metric.derive_quality(y)
    median, scale = M.geometry_stats(ds.geom)
    geom_z = ds.standardise(median, scale)
    cfg = M.ProbeConfig(n_geom=ds.geom.shape[1])
    paths = [args.data_dir / "train" / f"{i}.png" for i in ds.item_ids]
    n = len(ds)
    print(f"[ft] {n} items, device {device}, backbone lr {args.lr_backbone}, head lr {args.lr_head}", flush=True)

    model = FineTuneNet(cfg, args.model, grad_ckpt=args.grad_ckpt).to(device)

    if args.refit:
        train_idx = np.arange(n)
        val_idx = None
    elif args.val_fold is not None:
        val_idx = M.folds(n, 5, args.seed)[args.val_fold]
        train_idx = np.setdiff1d(np.arange(n), val_idx)
    else:
        raise SystemExit("specify --val-fold N or --refit")

    model, history, val_probs = train(
        model, ds, geom_z, paths, train_idx, device,
        epochs=args.epochs, batch=args.batch, lr_backbone=args.lr_backbone,
        lr_head=args.lr_head, wd=args.wd, seed=args.seed, val_idx=val_idx,
    )

    blob = {
        "backbone": model.backbone.state_dict(),
        "probe": model.probe.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "geom_median": median, "geom_scale": scale,
        "model_key": args.model,
        "recipe": {"epochs": args.epochs, "batch": args.batch, "lr_backbone": args.lr_backbone,
                   "lr_head": args.lr_head, "wd": args.wd, "seed": args.seed,
                   "grad_ckpt": args.grad_ckpt, "val_fold": args.val_fold, "refit": args.refit},
        "code_rev": git_rev(),
    }
    torch.save(blob, out / "ft.pt")

    if val_probs is not None:
        np.save(out / "oof_ft.npy", val_probs)
        np.save(out / f"val_idx_{args.val_fold}.npy", val_idx)
        res = metric.tune_thresholds(val_probs, y[val_idx], yq[val_idx])[1]
        print(f"[ft] fold-{args.val_fold} tuned score {res['score']:.3f} "
              f"(art {res['artefact_f1_weighted']:.3f}, qual {res['quality_f1']:.3f}); "
              f"baseline probe-s fold-0 reference: 13.730", flush=True)
        (out / "summary.json").write_text(json.dumps(
            {"score": res["score"], "artefact": res["artefact_f1_weighted"],
             "quality": res["quality_f1"],
             "per_label": res["per_label"], "history": history}, indent=2))

    if args.limit is not None:
        print("[ft] --limit run: smoke only, no test predict", flush=True)
        return 0

    _predict_test(model, ds, median, scale, out, device, args)
    return 0


def _predict_test(model, ds_tr, median, scale, out: Path, device, args) -> None:
    """Predict the test split with a trained FineTuneNet (optionally hflip TTA)."""
    ds_te = M.load_dataset(args.cache_dir, args.data_dir, model.model_key, "test", limit=args.limit)
    geom_z_te = ds_te.standardise(median, scale)
    paths_te = [args.data_dir / "test" / f"{i}.png" for i in ds_te.item_ids]
    probs = predict(model, ds_te, geom_z_te, paths_te, np.arange(len(ds_te)), device,
                    tta_flip=bool(args.tta))
    np.save(out / "probs_test_ft.npy", probs)
    np.save(out / "item_ids_test.npy", np.asarray(ds_te.item_ids))
    pos = (probs >= 0.5).sum(axis=0)
    print(f"[ft] test probs (tta_flip={bool(args.tta)}) -> {out / 'probs_test_ft.npy'}; "
          f"positives@0.5: {pos.tolist()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
