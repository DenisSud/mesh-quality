#!/usr/bin/env python
"""Profile the mesh tokenizer on a stratified sample of the real corpus.

    devenv shell -- uv run python main/mesh-quality/tools/profile_tokens.py --n 24

Reports per-stage timings, shape/finiteness checks and the spread of every
patch statistic, then extrapolates to a full-corpus cache build.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR / "src"))

from mesh_quality import mesh_tokens as mt  # noqa: E402


def stratified_paths(data_dir: Path, split: str, n: int, seed: int = 0) -> list[Path]:
    paths = sorted((data_dir / split).glob("*.npz"))
    sizes = np.array([p.stat().st_size for p in paths])
    order = np.argsort(sizes)
    rng = np.random.default_rng(seed)
    picks = set(np.round(np.geomspace(1, len(paths) - 1, max(n // 2, 2))).astype(int).tolist())
    picks |= set(rng.integers(0, len(paths), n).tolist())      # uniform random (median-heavy)
    picks |= {0, 1, len(paths) - 1, len(paths) - 2}             # extremes
    return [paths[order[i]] for i in sorted(picks)]


def sphere_mesh(n_lat: int = 64, n_lon: int = 128, radius: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere with a known area -- analytic tokenizer sanity check."""
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(-1, 3) * radius
    idx = np.arange((n_lat + 1) * n_lon).reshape(n_lat + 1, n_lon)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    f = np.concatenate([np.stack([a, b, d], -1), np.stack([b, c, d], -1)]).reshape(-1, 3)
    return v, f.astype(np.int32)


def stage_times(path: Path, cfg: mt.TokenConfig, seed: int) -> dict[str, float]:
    t = {}
    t0 = time.perf_counter()
    vertices, faces = mt.load_mesh(path)
    t["load"] = time.perf_counter() - t0
    vertices, faces, meta = mt.normalize_mesh(vertices, faces)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    sample = mt.sample_surface(vertices, faces, cfg.n_sample, rng)
    t["sample"] = time.perf_counter() - t0
    pts = sample["points"]

    t0 = time.perf_counter()
    centers = pts[mt.fps(pts, cfg.n_patch)]
    t["fps"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    members, radius = mt.knn_patches(centers, pts, cfg.k_points)
    t["knn"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bnd, nonman, bnd_frac, nonman_frac, has_struct = mt.census(vertices, faces, cfg)
    t["census"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = mt.patch_stats(
        vertices, faces, centers, radius, cfg, bnd, nonman, has_struct, float(sample["area_total"])
    )
    t["stats"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vox = mt.voxel_descriptor(vertices, faces, cfg.voxel_res)
    t["voxel"] = time.perf_counter() - t0
    return t, stats, vox, has_struct, members.shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=TASK_DIR / "data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--check-sphere", action="store_true")
    args = ap.parse_args()

    cfg = mt.TokenConfig()
    if args.check_sphere:
        v, f = sphere_mesh()
        tmp = Path("/tmp/token_sphere.npz")
        np.savez(tmp, vertices=v.astype(np.float32), faces=f.astype(np.int32))
        item = mt.tokenize_item(tmp, cfg, seed=1)
        area_n = item["globals"][3]
        true_area = 4 * np.pi * 0.25 / np.linalg.norm(np.array([1.0, 1.0, 1.0])) ** 2  # r=0.5, diag=sqrt(3)
        print(f"sphere: faces={len(f)} area_norm={np.exp(area_n):.6f} (analytic {true_area:.6f}) "
              f"valid={item['valid'].sum()}/{cfg.n_patch} "
              f"density_p50={np.median(item['stats'][:, 2]):.2f} (expect ~{np.log10(len(f)/true_area):.2f}) "
              f"radius_p50={np.exp(np.median(item['stats'][:, 11])):.4f} "
              f"normal_var_p50={np.median(item['stats'][:, 6]):.3f}")
        return 0

    paths = stratified_paths(args.data_dir, args.split, args.n)
    print(f"{len(paths)} meshes from {args.split} (size range {[p.stat().st_size // 1024 for p in paths[:2]]}.."
          f"{[p.stat().st_size // 1024 for p in paths[-2:]]} KiB)")

    totals = {k: 0.0 for k in ("load", "sample", "fps", "knn", "census", "stats", "voxel")}
    stats_rows, feat_lens, radii, vox_fracs, no_struct = [], [], [], [], 0
    for i, path in enumerate(paths, 1):
        t0 = time.perf_counter()
        try:
            times, stats, vox, has_struct, member_shape = stage_times(path, cfg, mt.item_seed(path.stem))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {path.stem}: {type(exc).__name__}: {exc}")
            continue
        wall = time.perf_counter() - t0
        for k, v in times.items():
            totals[k] += v
        stats_rows.append(stats)
        vox_fracs.append(float((vox > 0).mean()))
        no_struct += not has_struct
        if i <= 2 or path.stat().st_size > 3e7:
            print(f"  [{i:2d}] {path.stem[:8]} {path.stat().st_size/1e6:7.2f} MB  wall {wall:5.2f}s "
                  f"(load {times['load']:.2f} sample {times['sample']:.2f} fps {times['fps']:.2f} "
                  f"knn {times['knn']:.2f} census {times['census']:.2f} stats {times['stats']:.2f} "
                  f"vox {times['voxel']:.2f})  struct={has_struct} patch_members={member_shape}")

    total = sum(totals.values())
    n = max(len(stats_rows), 1)
    print(f"\nmean wall/item: {total/max(n,1):.2f}s  (load {totals['load']/n:.2f} sample {totals['sample']/n:.2f} "
          f"fps {totals['fps']/n:.2f} knn {totals['knn']/n:.2f} census {totals['census']/n:.2f} "
          f"stats {totals['stats']/n:.2f} voxel {totals['voxel']/n:.2f})")
    print(f"corpus estimate: {total/max(n,1)*9633/60:.1f} min single-core, /8 workers {total/max(n,1)*9633/480:.1f} min")
    print(f"census skipped: {no_struct}/{n} items; mean voxel occupancy {np.mean(vox_fracs):.3f}")

    allstats = np.concatenate(stats_rows)
    print(f"\npatch stats over {len(allstats)} patches: min / p1 / p50 / p99 / max, and fraction of exact zeros")
    for j, name in enumerate(mt.STAT_NAMES):
        col = allstats[:, j]
        q = np.percentile(col, [1, 50, 99])
        print(f"  {name:24s} {col.min():10.3f} {q[0]:10.3f} {q[1]:10.3f} {q[2]:10.3f} {col.max():10.3f} "
              f" zero={np.mean(col == 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
