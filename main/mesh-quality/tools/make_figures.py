#!/usr/bin/env python
"""Figures for the mesh-quality presentation and notebooks.

    uv run python main/mesh-quality/tools/make_figures.py
    uv run python main/mesh-quality/tools/make_figures.py --attention <item_id>

Everything is CPU-only except ``--attention``, which runs the gradient
attribution of ``mesh_quality.visualize`` for one item (needs the DINOv3 weights
and, comfortably, a GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
CACHE = TASK / "cache"
OUT = TASK / "presentation" / "figures"

from mesh_quality import metric  # noqa: E402

#: model key -> label used in the figures
SERIES = {
    "geometry": "геометрия\n(HistGB)",
    "s": "DINOv3-s\n+ проб",
    "b": "DINOv3-b\n+ проб",
    "fuse": "ансамбль\ns+b",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _train_labels():
    labels = pd.read_csv(TASK / "data" / "train.csv").set_index("item_id")
    ids = np.asarray(labels.index.astype(str))
    y = labels[list(metric.DEFECTS)].to_numpy(dtype=np.int8)
    return ids, y, labels


def _scores() -> dict[str, dict]:
    """OOF score for every model we have probabilities for."""
    out: dict[str, dict] = {}
    candidates = {k: CACHE / f"oof_probe_{k}.npy" for k in ("s", "b")}
    candidates["geometry"] = CACHE / "oof_geometry.npy"
    for key, path in candidates.items():
        if path.exists():
            out[key] = {"oof": np.load(path).astype(np.float32)}
    fuses = sorted(CACHE.glob("oof_fuse_*.npy"))
    if fuses:
        out["fuse"] = {"oof": np.load(fuses[-1]).astype(np.float32), "name": fuses[-1].stem}
    return out


def fig_labels(plt) -> None:
    _, y, _ = _train_labels()
    counts = y.sum(0)
    quality = metric.derive_quality(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.1, 1]})
    order = np.argsort(counts)[::-1]
    names = [metric.DEFECTS[i] for i in order]
    vals = counts[order]
    ax = axes[0]
    bars = ax.bar(names, vals, color="#3b6ea5")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 40, f"{v}\n{100 * v / len(y):.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Частота дефектов в обучающей выборке (n={len(y)})")
    ax.set_ylabel("объектов")
    ax.set_ylim(0, vals.max() * 1.28)
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1]
    q1 = int(quality.sum())
    ax.bar(["quality = 1", "quality = 0"], [q1, len(y) - q1], color=["#4c9f70", "#b4553c"])
    for i, v in enumerate([q1, len(y) - q1]):
        ax.text(i, v + 60, f"{v}\n{100 * v / len(y):.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Мишень quality")
    ax.set_ylim(0, max(q1, len(y) - q1) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_labels.png", bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig_labels.png")


def fig_results(plt) -> None:
    scores = _scores()
    _, y, _ = _train_labels()
    yq = metric.derive_quality(y)
    names, values, quality_f1, art_f1 = [], [], [], []
    for key, payload in scores.items():
        thr, res = metric.tune_thresholds(payload["oof"], y, yq, verbose=False)
        names.append(SERIES.get(key, key))
        values.append(res["score"])
        quality_f1.append(res["quality_f1"])
        art_f1.append(res["artefact_f1_weighted"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    colors = ["#9aa7b1", "#3b6ea5", "#2f8f6f", "#8a5cc4"][: len(names)]
    bars = ax.bar(names, values, color=colors)
    for bar, v, q, a in zip(bars, values, quality_f1, art_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, f"{10 * q:.1f} +\n{10 * a:.1f}",
                ha="center", va="center", fontsize=8, color="white")
    ax.set_ylabel("баллы (OOF, из 20)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("Счёт на кросс-валидации (5 фолдов, 8964 объекта)")
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[1]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [10 * v for v in quality_f1], width=0.38, label="10 · F1(quality)", color="#4c9f70")
    ax.bar(x + 0.2, [10 * v for v in art_f1], width=0.38, label="10 · F1w(artefacts)", color="#b4553c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("баллы")
    ax.set_title("Из чего состоит счёт")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_results.png", bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig_results.png")


def fig_perlabel(plt) -> None:
    scores = _scores()
    _, y, _ = _train_labels()
    yq = metric.derive_quality(y)
    per = {}
    for key, payload in scores.items():
        thr, res = metric.tune_thresholds(payload["oof"], y, yq, verbose=False)
        per[key] = res["per_label"]
    keys = [k for k in ("geometry", "s", "b", "fuse") if k in per]
    x = np.arange(len(metric.DEFECTS))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    order = np.argsort([per[keys[0]][n] for n in metric.DEFECTS])  # by geometry, ascending
    names = [metric.DEFECTS[i] for i in order]
    palette = {"geometry": "#9aa7b1", "s": "#3b6ea5", "b": "#2f8f6f", "fuse": "#8a5cc4"}
    for j, key in enumerate(keys):
        vals = [per[key][n] for n in names]
        ax.bar(x + (j - (len(keys) - 1) / 2) * width, vals, width=width * 0.92,
               label=SERIES.get(key, key).replace("\n", " "), color=palette.get(key, "#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 по классам на OOF: что добавляют изображения")
    ax.legend(frameon=False, fontsize=9, ncol=len(keys))
    fig.tight_layout()
    fig.savefig(OUT / "fig_perlabel.png", bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig_perlabel.png")


def fig_thresholds(plt) -> None:
    path = CACHE / "oof_probe_s.npy"
    _, y, _ = _train_labels()
    probs = np.load(path).astype(np.float32)
    grid = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(1, len(("noisy", "artifacts", "open")), figsize=(11, 3.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), ("noisy", "artifacts", "open")):
        k = metric.DEFECTS.index(name)
        f1 = []
        for t in grid:
            pred = (probs[:, k] >= t).astype(np.int8)
            tp = int(((pred == 1) & (y[:, k] == 1)).sum())
            fp = int(((pred == 1) & (y[:, k] == 0)).sum())
            fn = int(((pred == 0) & (y[:, k] == 1)).sum())
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        f1 = np.asarray(f1)
        best = grid[int(f1.argmax())]
        ax.plot(grid, f1, color="#3b6ea5", lw=2)
        ax.axvline(best, color="#b4553c", ls="--", lw=1.2, label=f"порог {best:.2f} (F1 {f1.max():.2f})")
        ax.set_title(f"{name}  (доля класса {100 * y[:, k].mean():.0f}%)")
        ax.set_xlabel("порог вероятности")
        ax.legend(frameon=False, fontsize=8, loc="lower center")
    np.atleast_1d(axes)[0].set_ylabel("F1")
    fig.suptitle("Порог решает всё: F1(класса) как функция порога (OOF, DINOv3-s + проб)", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_thresholds.png", bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig_thresholds.png")


def fig_resolution(plt, item_id: str | None = None) -> None:
    """Show what patch pooling means: one token covers a 64x64 px region."""
    from PIL import Image

    if item_id is None:
        # a clean (quality = 1), well-recognizable train object without render text
        item_id = "13c6c9de-8e12-4010-a96d-c8543a5c4417"
    path = TASK / "data" / "train" / f"{item_id}.png"
    img = np.asarray(Image.open(path).convert("RGB"))
    tile = img[:512, :512]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    axes[0].imshow(tile)
    axes[0].set_title("тайл 512×512 (az0)", fontsize=10)
    axes[0].axis("off")
    for ax, g, colour in ((axes[1], 8, "#ffd166"), (axes[2], 16, "#8ecae6")):
        ax.imshow(tile)
        step = 512 / g
        for i in range(g + 1):
            ax.axhline(i * step, color=colour, lw=0.6, alpha=0.9)
            ax.axvline(i * step, color=colour, lw=0.6, alpha=0.9)
        ax.set_title(f"сетка {g}×{g}: токен = {int(512 / g)}×{int(512 / g)} px\n({g * g} токенов на тайл)", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_resolution.png", bbox_inches="tight")
    plt.close(fig)
    print("[figures] fig_resolution.png")


def fig_attention(item_id: str) -> None:
    """Run the attribution tool for one item and collect its contact sheets."""
    from argparse import Namespace

    from mesh_quality import visualize

    out_dir = OUT / "attribution"
    args = Namespace(
        task_dir=TASK,
        data_dir=TASK / "data",
        model="s",
        split="test",
        item_id=item_id,
        labels=None,
        device=None,
        out_dir=out_dir,
    )
    visualize.run(args)
    print(f"[figures] attribution sheets in {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attention", default=None, metavar="ITEM_ID", help="also render attribution sheets")
    ap.add_argument("--skip-heavy", action="store_true", help="skip the resolution figure")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    plt = _style()
    fig_labels(plt)
    fig_results(plt)
    fig_perlabel(plt)
    fig_thresholds(plt)
    if not args.skip_heavy:
        fig_resolution(plt)
    if args.attention:
        fig_attention(args.attention)
    print(f"[figures] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
