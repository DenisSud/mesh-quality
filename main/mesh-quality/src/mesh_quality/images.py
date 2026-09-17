"""Image side: render tiles and frozen DINOv3 patch features.

The renders are collages of six 512x512 views of the same object, row-major:
four azimuths 90 degrees apart, then top and bottom.  Each tile goes through a
frozen DINOv3 ViT (patch 16) at native 512x512 resolution, giving a 32x32 grid
of patch tokens per tile.  Extraction pools each tile's grid 4x4 -> 8x8 (one
token per 64x64 px region) and caches the pooled grid plus the two tile-level
summaries, so the probe never needs the 27 GB of raw data again:

- ``tile_mean`` / ``tile_max``: pooled over the whole tile (both are kept: mean
  blurs thin structures such as annotation strokes, max keeps them),
- ``tile_grid``: the 8x8 pooled patch grid (the probe consumes it directly).

One feature cache = one directory per backbone x split::

    cache/dino{s,b}_{train,test}/
        meta.json      grid_size, dim, n_items, ...
        item_ids.npy
        tile_mean.npy  [n, 6, dim]          fp16
        tile_max.npy   [n, 6, dim]          fp16
        tile_grid.npy  [n, 6, 8, 8, dim]    fp16

Run the extraction through the task CLI::

    python main/mesh-quality/main.py features --model s
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TILE_PX = 512
PATCH = 16
GRID = TILE_PX // PATCH  # 32 native patch tokens per tile side
POOLED_GRID = 8  # probe token grid per tile: 4x4 block mean of the native grid
N_TILES = 6

#: Short key -> timm model name.  All are patch-16 DINOv3 students and accept
#: 512x512 input directly (RoPE, so no positional interpolation is needed).
MODELS: dict[str, str] = {
    "s": "vit_small_plus_patch16_dinov3",  # 28.7M, d=384
    "b": "vit_base_patch16_dinov3",  # 85.6M, d=768
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureCache:
    """One split's cached probe inputs."""

    item_ids: np.ndarray
    tile_mean: np.ndarray  # [n, 6, dim] fp16
    tile_max: np.ndarray
    tile_grid: np.ndarray  # [n, 6, POOLED_GRID, POOLED_GRID, dim] fp16
    grid: int  # pooled patch grid stored on disk
    dim: int  # backbone feature width


def split_tiles(path: str | Path) -> np.ndarray:
    """Load a ``.png`` collage and split it into ``[6, 512, 512, 3]`` uint8 tiles."""
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    expected = (2 * TILE_PX, 3 * TILE_PX)
    if (h, w) != expected:
        raise ValueError(f"{path}: expected {expected} collage, got {(h, w)}")
    rows = [
        img[(i // 3) * TILE_PX : (i // 3 + 1) * TILE_PX,
            (i % 3) * TILE_PX : (i % 3 + 1) * TILE_PX]
        for i in range(N_TILES)
    ]
    return np.stack(rows)


def cache_path(cache_dir: str | Path, model_key: str, split: str) -> Path:
    """Directory holding one backbone x split cache."""
    return Path(cache_dir) / f"dino{model_key}_{split}"


def _load_model(model_key: str, device: str):
    import timm

    if model_key not in MODELS:
        raise KeyError(f"unknown model key {model_key!r}, expected one of {sorted(MODELS)}")
    model = timm.create_model(MODELS[model_key], pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _normalise(tiles_u8: np.ndarray, device: str):
    """``[B, 512, 512, 3]`` uint8 -> normalised ``[B, 3, 512, 512]`` on ``device``.

    Only the uint8 batch crosses PCIe; the float conversion and normalisation run
    on the GPU (on the host they cost ~450 MB of strided fp32 traffic per batch).
    """
    import torch

    x = torch.from_numpy(np.ascontiguousarray(tiles_u8)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _decoded_tiles(paths, workers: int = 4, window: int = 48):
    """Yield decoded collages in order, decoding ahead on ``workers`` threads.

    PNG decode is ~15 ms per item and would otherwise leave the GPU idle between
    batches (PIL releases the GIL, so threads are enough).
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(paths)
        pending = []
        for p in it:
            pending.append(pool.submit(split_tiles, p))
            if len(pending) >= window:
                break
        while pending:
            yield pending.pop(0).result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(split_tiles, nxt))


def pool_patches(patches):
    """``[..., 32, 32, D]`` native patch grid -> ``[..., 8, 8, D]`` block means.

    Average-pools disjoint 4x4 blocks in fp32: this is the one owner of the
    "which patches become probe tokens" decision, shared by extraction, the
    attribution tools and the tests.  Accepts any leading batch shape and works
    on CPU or CUDA.
    """
    import torch.nn.functional as F

    lead = patches.shape[:-3]
    if patches.shape[-3] != GRID or patches.shape[-2] != GRID:
        raise ValueError(f"pool_patches expects a {GRID}x{GRID} patch grid, got {tuple(patches.shape[-3:])}")
    x = patches.reshape(-1, GRID, GRID, patches.shape[-1]).float()
    x = F.avg_pool2d(x.permute(0, 3, 1, 2), GRID // POOLED_GRID).permute(0, 2, 3, 1)
    return x.reshape(*lead, POOLED_GRID, POOLED_GRID, patches.shape[-1])


def _tile_summaries(patches):
    """``[B, 32, 32, D]`` patch tokens -> (tile_mean, tile_max, pooled grid)."""
    flat = patches.flatten(1, 2)  # [B, 1024, D]
    return flat.mean(1), flat.max(1).values, pool_patches(patches)


def extract_split(
    data_dir: str | Path,
    split: str,
    cache_dir: str | Path,
    model_key: str = "s",
    batch_tiles: int = 144,
    limit: int | None = None,
    device: str | None = None,
    log=print,
) -> Path:
    """Extract frozen DINOv3 features for one split; returns the cache directory.

    The pooled grid and the tile summaries are written per item as they arrive,
    so nothing keeps the raw data around after the split is done.  ``limit`` is
    for smoke tests only.
    """
    import pandas as pd
    import torch

    data_dir = Path(data_dir)
    ids = pd.read_csv(data_dir / f"{split}.csv")["item_id"].tolist()
    if limit is not None:
        ids = ids[:limit]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_key, device)
    dim = int(model.embed_dim)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    n_patch = GRID * GRID

    n = len(ids)
    dest = cache_path(cache_dir, model_key, split)
    dest.mkdir(parents=True, exist_ok=True)
    tile_mean = np.zeros((n, N_TILES, dim), dtype=np.float16)
    tile_max = np.zeros((n, N_TILES, dim), dtype=np.float16)
    tile_grid = np.zeros((n, N_TILES, POOLED_GRID, POOLED_GRID, dim), dtype=np.float16)

    buf: list[np.ndarray] = []
    owners: list[tuple[int, int]] = []
    t0 = time.time()

    def flush():
        if not buf:
            return
        tiles = np.stack(buf)  # [B, 512, 512, 3]
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            feats = model.forward_features(_normalise(tiles, device))
        patches = feats[:, n_prefix : n_prefix + n_patch].float()
        patches = patches.reshape(len(tiles), GRID, GRID, dim)
        mean_t, max_t, grid_t = _tile_summaries(patches)
        mean_a = mean_t.cpu().numpy()
        max_a = max_t.cpu().numpy()
        grid_a = grid_t.cpu().numpy()
        for k, (item_i, tile_i) in enumerate(owners):
            tile_mean[item_i, tile_i] = mean_a[k]
            tile_max[item_i, tile_i] = max_a[k]
            tile_grid[item_i, tile_i] = grid_a[k]
        buf.clear()
        owners.clear()

    for i, tiles in enumerate(_decoded_tiles(data_dir / split / f"{item_id}.png" for item_id in ids)):
        for t in range(N_TILES):
            buf.append(tiles[t])
            owners.append((i, t))
        if len(buf) >= batch_tiles:
            flush()
        if (i + 1) % 200 == 0 or i + 1 == n:
            done = i + 1
            rate = done / max(time.time() - t0, 1e-9)
            log(
                f"  {split} {done}/{n} items ({rate:.1f} item/s, "
                f"eta {((n - done) / max(rate, 1e-9)) / 60:.1f} min)"
            )
    flush()

    np.save(dest / "tile_mean.npy", tile_mean)
    np.save(dest / "tile_max.npy", tile_max)
    np.save(dest / "tile_grid.npy", tile_grid)
    np.save(dest / "item_ids.npy", np.array(ids))
    meta = {
        "model_key": model_key,
        "model": MODELS[model_key],
        "split": split,
        "grid_size": POOLED_GRID,
        "native_grid_size": GRID,
        "dim": dim,
        "n_tiles": N_TILES,
        "n_items": n,
        "dtype": "float16",
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    log(f"  wrote {dest} ({n} items) in {(time.time() - t0) / 60:.1f} min")
    return dest


def load_cache(cache_dir: str | Path, model_key: str, split: str) -> FeatureCache:
    """Open a feature cache written by :func:`extract_split`."""
    d = cache_path(cache_dir, model_key, split)
    if not (d / "meta.json").exists():
        raise SystemExit(f"missing feature cache {d} -- run `main.py features --model {model_key}` first")
    meta = json.loads((d / "meta.json").read_text())
    return FeatureCache(
        item_ids=np.load(d / "item_ids.npy"),
        tile_mean=np.load(d / "tile_mean.npy"),
        tile_max=np.load(d / "tile_max.npy"),
        tile_grid=np.load(d / "tile_grid.npy"),
        grid=int(meta["grid_size"]),
        dim=int(meta["dim"]),
    )
