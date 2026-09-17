"""L4 submission path: refit the joint mesh probe on ALL train rows, predict test.

Mirrors ``solution.py``'s full-data refit convention (same recipe as the CV run,
seed 0, single state) but through the joint mesh factory from ``tools/probe_mesh.py``
so gradients reach the JEPA encoder.  The checkpoint stores everything needed to
rebuild the model (probe cfg, JepaConfig, geometry stats, code revision) -- the
provenance lesson from ``knowledge/mesh-quality-solution.md`` §8.

    devenv shell -- uv run python main/mesh-quality/tools/refit_mesh.py \
        --ckpt main/mesh-quality/cache/jepa_pretrain/run1/ckpt_e275.pt \
        --out-dir main/mesh-quality/cache/l4_full

Smoke (CPU, ~2 min):   --limit 64 --epochs 1 --batch 8 --device cpu
Resume predict only:   --predict-only (uses the saved refit checkpoint)
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
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "tools"))

from mesh_quality import jepa, model as M  # noqa: E402
from probe_mesh import (  # noqa: E402
    JointMeshProbe,
    RawBatchDataset,
    make_joint_factory,
    mesh_probe,
    tokeniser_order,
)


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_split(cache_dir: Path, data_dir: Path, model_key: str, split: str,
                batch: int, device: torch.device, limit: int | None):
    """Dataset + raw-batch mesh wrapper for one split (probe order)."""
    ds = M.load_dataset(cache_dir, data_dir, model_key, split, limit=limit)
    order = tokeniser_order(None, split, ds.item_ids, cache_dir)
    batcher = jepa.MemmapBatcher(cache_dir / "mesh_tokens", [split], batch, device, shuffle=False)
    mesh_ds = RawBatchDataset(ds, batcher, order, torch.from_numpy(order >= 0))
    return ds, mesh_ds, int((order < 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache")
    ap.add_argument("--data-dir", type=Path, default=TASK / "data")
    ap.add_argument("--model", default="s", help="DINOv3 cache key (L4 recipe: s)")
    ap.add_argument("--ckpt", type=Path, required=True, help="JEPA checkpoint (warm start)")
    ap.add_argument("--out-dir", type=Path, default=TASK / "cache" / "l4_full")
    ap.add_argument("--epochs", type=int, default=M.TrainRecipe.epochs)
    ap.add_argument("--batch", type=int, default=96, help="L4 CV ran at 96")
    ap.add_argument("--lr", type=float, default=M.TrainRecipe.lr)
    ap.add_argument("--wd", type=float, default=M.TrainRecipe.wd)
    ap.add_argument("--seed", type=int, default=M.TrainRecipe.seed)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="with --limit: still save + run the test-predict path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = out / "refit_l4.pt"

    if args.predict_only and not ckpt_path.exists():
        raise SystemExit(f"--predict-only: {ckpt_path} not found")

    t0 = time.time()
    if not args.predict_only:
        ds, mesh_ds, missing = build_split(args.cache_dir, args.data_dir, args.model, "train",
                                           args.batch, device, args.limit)
        y = np.asarray(ds.labels)
        print(f"[refit] train {len(ds)} items ({missing} without mesh tokens), device {device}", flush=True)
        median, scale = M.geometry_stats(ds.geom)
        geom_z = ds.standardise(median, scale)
        cfg = M.ProbeConfig(n_geom=ds.geom.shape[1])
        encoder, jcfg = jepa.load_checkpoint(args.ckpt, device="cpu")
        factory = make_joint_factory(encoder.state_dict(), jcfg, null_embed=True)
        with mesh_probe(factory=factory):
            net, hist = M.train_model(
                mesh_ds, cfg, geom_z, np.arange(len(ds)),
                epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd,
                seed=args.seed, device=str(device), log_every=10,
            )
        if args.limit is not None and not args.smoke:
            print("[refit] --limit run: smoke test only, nothing written", flush=True)
            return 0
        blob = {
            "state": net.state_dict(),
            "cfg": dataclasses.asdict(cfg),
            "jcfg": dataclasses.asdict(jcfg),
            "geom_median": median,
            "geom_scale": scale,
            "model_key": args.model,
            "recipe": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                       "wd": args.wd, "seed": args.seed},
            "jepa_ckpt": str(args.ckpt),
            "code_rev": git_rev(),
        }
        torch.save(blob, ckpt_path)
        print(f"[refit] final loss {hist[-1]:.4f}; checkpoint {ckpt_path.name} "
              f"({ckpt_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)

    # ---- test prediction with the refit state ----
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = M.ProbeConfig(**{k: v for k, v in blob["cfg"].items()})
    jcfg = jepa.JepaConfig(**blob["jcfg"])
    ds_te, mesh_te, missing = build_split(args.cache_dir, args.data_dir, blob["model_key"], "test",
                                          args.batch, device, args.limit)
    geom_z_te = ds_te.standardise(np.asarray(blob["geom_median"]), np.asarray(blob["geom_scale"]))
    print(f"[refit] test {len(ds_te)} items ({missing} without mesh tokens)", flush=True)
    with mesh_probe():
        net = JointMeshProbe(cfg, encoder=jepa.MeshJepa(jcfg), null_embed=True)
        net.load_state_dict(blob["state"])
        net.to(device)
        probs = M.predict_probs(net, mesh_te, geom_z_te, np.arange(len(ds_te)), device=str(device))
    np.save(out / "probs_test_l4.npy", probs)
    np.save(out / "item_ids_test.npy", np.asarray(ds_te.item_ids))
    pos = (probs >= 0.5).sum(axis=0)
    print(f"[refit] test probs -> {out / 'probs_test_l4.npy'}; positives@0.5: {pos.tolist()}", flush=True)
    print(f"[refit] done in {time.time() - t0:.0f}s (code_rev {blob.get('code_rev')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
