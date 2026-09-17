"""Per-label patch attribution for a single item.

The probe consumes *pooled* DINOv3 features, so attribution is done on the
native 32x32 patch grid recomputed on the fly for one item: we run the backbone,
pool the grid to the probe's 8x8 cells, feed the probe, backpropagate each
defect logit to the cell inputs and take ``|grad x activation|`` per cell and
per tile.

Outputs (into ``--out-dir``): one PNG per requested label with the six render
tiles and the attribution heat map, plus a ``.json`` with the top cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import images, metric, model

HEAT_TILE = 256  # tiles are downscaled to this for the contact sheet


def _colourise(heat: np.ndarray) -> np.ndarray:
    """Map a [0,1] array to an RGB uint8 image (blue -> cyan -> yellow -> red)."""
    stops = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 1.0],
            [0.2, 0.9, 0.6],
            [1.0, 0.9, 0.2],
            [0.9, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    x = np.clip(heat, 0, 1) * (len(stops) - 1)
    lo = np.floor(x).astype(int).clip(0, len(stops) - 2)
    frac = (x - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return (rgb * 255).astype(np.uint8)


def _upsample(heat: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upsample of a cell-grid heat map to ``size`` pixels."""
    idx = (np.arange(size) * heat.shape[0] // size).clip(0, heat.shape[0] - 1)
    return heat[idx][:, idx]


def _overlay(tile: np.ndarray, heat: np.ndarray, floor: float, alpha: float = 0.9) -> np.ndarray:
    """Tint only the salient cells: the heat map drives the overlay opacity.

    Cells below ``floor`` leave the render untouched; above it both the colour
    ramp and the opacity grow.  ``floor`` is chosen per item (see
    :func:`_contact_sheet`) so the sheet highlights a handful of hot cells
    instead of washing out the whole tile.
    """
    h = _upsample(heat, tile.shape[0])
    colour = _colourise(h).astype(np.float32)
    opaque = np.clip((h - floor) / (1.0 - floor + 1e-9), 0.0, 1.0) ** 0.6 * alpha
    base = tile.astype(np.float32)
    return (base * (1 - opaque[..., None]) + colour * opaque[..., None]).astype(np.uint8)


def _contact_sheet(tiles: np.ndarray, heats: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    from PIL import Image

    # Keep only the top ~15% of cells: attribution is sparse, so a global floor
    # is more readable than a full ramp over all 6 x 8 x 8 cells.
    floor = float(np.quantile(np.asarray(heats).reshape(-1), 0.94))
    rows = []
    for r in range(2):
        row = []
        for c in range(3):
            k = r * 3 + c
            tile = np.asarray(
                Image.fromarray(tiles[k]).resize((HEAT_TILE, HEAT_TILE), Image.BILINEAR), dtype=np.uint8
            )
            row.append(_overlay(tile, heats[k], floor=floor, alpha=alpha))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _full_grid_features(item_path: Path, device: str, model_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the native 32x32 patch grid for one item; returns (tokens, tiles)."""
    import torch

    tiles = images.split_tiles(item_path)
    net = images._load_model(model_key, device)
    with torch.inference_mode():
        x = images._normalise(tiles, device)
        feats = net.forward_features(x)
        n_prefix = int(getattr(net, "num_prefix_tokens", 1))
        tokens = feats[:, n_prefix:].float().cpu().numpy()
    del net
    torch.cuda.empty_cache()
    return tokens, tiles  # [6, 1024, D], [6, 512, 512, 3]


def run(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    cache_dir = args.task_dir / "cache"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cache_dir / f"probe_{args.model}.pt", map_location=device, weights_only=False)
    cfg = model.ProbeConfig(**ckpt["cfg"])
    net = model.AttentiveProbe(cfg).to(device).eval()
    net.load_state_dict(ckpt["states"][0])

    split = args.split
    item_path = Path(args.data_dir) / split / f"{args.item_id}.png"
    if not item_path.exists():
        raise SystemExit(f"{item_path} not found")

    tokens, tiles = _full_grid_features(item_path, device, ckpt["model_key"])
    g = cfg.grid
    dim = tokens.shape[-1]
    grid_tokens = images.pool_patches(
        torch.from_numpy(tokens.reshape(images.N_TILES, images.GRID, images.GRID, dim))
    ).numpy()
    tile_mean = tokens.mean(axis=1)
    tile_max = tokens.max(axis=1)

    # Geometry block: the same 27 columns (23 raw + 4 derived) as in training,
    # aligned through the cached table so the column order always matches.
    geom, geom_names = model._geometry_table(cache_dir, split, np.asarray([args.item_id]))
    if geom.shape[1] != cfg.n_geom:
        raise SystemExit(f"geometry has {geom.shape[1]} columns, probe expects {cfg.n_geom} ({geom_names})")

    median = np.asarray(ckpt["geom_median"])
    scale = np.asarray(ckpt["geom_scale"])
    geom = model.standardise_geom(geom, median, scale).astype(np.float32)

    mean_t = torch.from_numpy(tile_mean[None].astype(np.float32)).to(device).requires_grad_(True)
    max_t = torch.from_numpy(tile_max[None].astype(np.float32)).to(device).requires_grad_(True)
    grid_t = torch.from_numpy(grid_tokens[None].astype(np.float32)).to(device).requires_grad_(True)
    geom_t = torch.from_numpy(geom).to(device)

    logits = net(mean_t, max_t, grid_t, geom_t)[0]
    thr = np.asarray(ckpt["thresholds"], dtype=np.float32)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    labels = args.labels.split(",") if args.labels else [n for n, p in zip(metric.DEFECTS, probs) if p >= 0.5]
    if not labels:
        labels = [metric.DEFECTS[int(np.argmax(probs))]]

    summary = {"item_id": args.item_id, "split": split, "labels": {}}
    for name in labels:
        k = metric.DEFECTS.index(name)
        net.zero_grad(set_to_none=True)
        if mean_t.grad is not None:
            mean_t.grad = None
        if max_t.grad is not None:
            max_t.grad = None
        grid_t.grad = None
        logits[k].backward(retain_graph=True)
        sal = (grid_t.grad * grid_t).abs().sum(-1)[0].detach().cpu().numpy()  # [6, g, g]
        sal = sal / (sal.max() + 1e-9)
        sheet = _contact_sheet(tiles, sal)
        png = out_dir / f"{args.item_id}_{name}.png"
        Image.fromarray(sheet).save(png)
        flat = np.argsort(sal.reshape(-1))[::-1][:5]
        cells = [
            {"tile": int(c // (g * g)), "row": int((c % (g * g)) // g), "col": int(c % g), "saliency": float(sal.reshape(-1)[c])}
            for c in flat
        ]
        summary["labels"][name] = {"probability": float(probs[k]), "threshold": float(thr[k]), "top_cells": cells}
        print(f"[visualize] {name}: p={probs[k]:.3f} thr={thr[k]:.2f} -> {png}")

    (out_dir / f"{args.item_id}.json").write_text(json.dumps(summary, indent=2))
    print(f"[visualize] wrote {out_dir / f'{args.item_id}.json'}")
