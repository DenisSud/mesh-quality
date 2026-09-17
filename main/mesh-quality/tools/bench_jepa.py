#!/usr/bin/env python
"""GPU throughput benchmark + end-to-end smoke run for the mesh-patch JEPA.

    devenv shell -- uv run python main/mesh-quality/tools/bench_jepa.py            # synthetic shapes
    devenv shell -- uv run python main/mesh-quality/tools/bench_jepa.py --real     # tokenizer cache

Reports ms/step, items/s and the implied epoch time for the planned config, plus
optional variants (amp, compile, matmul precision, batch size).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import jepa, mesh_tokens as mt  # noqa: E402


def fake_batch(b: int, k: int, device: torch.device, n_valid: int, m: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(0)
    valid = torch.zeros(b, k, dtype=torch.bool, device=device)
    valid[:, :n_valid] = True
    return {
        "pts": (torch.randn(b, k, 32, len(mt.POINT_NAMES), generator=gen, device=device) * 0.5).half(),
        "stats": torch.randn(b, k, len(mt.STAT_NAMES), generator=gen, device=device),
        "centers": torch.rand(b, k, 3, generator=gen, device=device) - 0.5,
        "valid": valid,
        "globals": torch.zeros(b, len(mt.GLOBAL_NAMES), device=device),
    }


def bench_step(
    model: jepa.MeshJepa, batch: dict, device: torch.device, steps: int = 12, warmup: int = 5
) -> tuple[float, float, float]:
    """Steady-state ms/step and items/s (plus the cold first step, for context)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator(device=device).manual_seed(0)
    model.train()

    def one_step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.cfg.amp):
            out = model(batch, generator=gen)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    one_step()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dt * 1000, batch["pts"].shape[0] / dt, cold_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use the tokenizer cache for shapes/validity")
    ap.add_argument("--cache-dir", type=Path, default=TASK / "cache" / "mesh_tokens")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", default="128,256,384")
    ap.add_argument("--items", type=int, default=8964)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} "
          f"torch={torch.__version__}", flush=True)
    torch.set_float32_matmul_precision("high")

    if args.real:
        ds = jepa.MeshPatchDataset(args.cache_dir, args.split)
        print(f"real cache: {len(ds)} items, valid frac {np.asarray(ds.valid).mean():.3f}")
        sample = [ds[i] for i in range(min(64, len(ds)))]
        n_valid = int(np.mean([int(s["valid"].sum()) for s in sample]))
        stats_hist = [int(s["valid"].sum()) for s in sample]
        print(f"valid patches per item: min {min(stats_hist)} p50 {int(np.median(stats_hist))} max {max(stats_hist)}")
    else:
        n_valid = 256

    for batch_size in [int(x) for x in args.batches.split(",")]:
        cfg = jepa.JepaConfig(batch=batch_size, amp=not args.fp32)
        model = jepa.MeshJepa(cfg).to(device)
        if args.compile:
            model.encoder = torch.compile(model.encoder, dynamic=True)
            model.predictor = torch.compile(model.predictor, dynamic=True)
        batch = fake_batch(batch_size, 256, device, n_valid=n_valid, m=0)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        ms, ips, cold_ms = bench_step(model, batch, device, args.steps, args.warmup)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        epoch_s = args.items / ips
        params = sum(p.numel() for p in model.parameters())
        print(f"batch {batch_size:4d} amp={not args.fp32} compile={args.compile}: {ms:7.1f} ms/step  "
              f"{ips:7.1f} items/s  epoch {epoch_s:5.1f}s  peak {peak:5.1f} GB  params {params/1e6:.2f}M  "
              f"(cold {cold_ms:.0f} ms)", flush=True)

    if args.real and device.type == "cuda":
        # dataloader throughput alone (page-cache reads + host->device)
        ds = jepa.MeshPatchDataset(args.cache_dir, args.split)
        idx = np.arange(len(ds))
        t0 = time.perf_counter()
        for start in range(0, 2048, 256):
            batch_idx = idx[start:start + 256]
            pts = torch.from_numpy(np.asarray(ds.pts[batch_idx], dtype=np.float32)).to(device)
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"loader: {2048 / dt:6.1f} items/s ({dt * 1000 / 8:.1f} ms/batch of 256) incl. H2D", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
