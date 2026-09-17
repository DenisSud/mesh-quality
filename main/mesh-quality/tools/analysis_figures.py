#!/usr/bin/env python
"""Analysis figures for the mesh-quality presentation and notebook.

CPU figures (no model, seconds):

    fig_data_mesh    mesh sizes, connectivity (unwelded components), label structure
    fig_cooc         conditional defect co-occurrence
    fig_errors       per-label FP/FN structure of the OOF predictions
    fig_curves       train/val curves from tools/hyperparams.py
    fig_hparams      ablation scores from tools/hyperparams.py

GPU figures (one frozen-DINOv3 forward per item, ~1 min each):

    fig_methods      gradient saliency vs occlusion vs query attention
    fig_views        leave-one-view-out logit change (6 views x 10 labels)
    fig_cases        curated failure/success cases with attribution sheets

Usage (from the repo root):

    devenv shell -- uv run python main/mesh-quality/tools/analysis_figures.py
    devenv shell -- uv run python main/mesh-quality/tools/analysis_figures.py --cpu-only
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
sys.path.insert(0, str(TASK / "tools"))
CACHE = TASK / "cache"
OUT = TASK / "presentation" / "figures"
ABL = CACHE / "ablations"
LABELS = ["abstract", "artifacts", "intersection", "lowpoly", "noisy", "open", "partial", "scale", "set", "simple"]
VIEWS = ["az0", "az90", "az180", "az270", "top", "bottom"]


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


def train_labels():
    """(item_ids, defect matrix, quality vector, geometry frame) for the train split."""
    lab = pd.read_csv(TASK / "data" / "train.csv").set_index("item_id")
    ids = np.asarray(lab.index.astype(str))
    y = lab[LABELS].to_numpy(dtype=np.int8)
    geo = pd.read_csv(CACHE / "geometry_train.csv").set_index("item_id").reindex(list(ids))
    return ids, y, (y.sum(1) == 0).astype(np.int8), geo, lab


# --------------------------------------------------------------------------- #
# CPU figures
# --------------------------------------------------------------------------- #


def fig_data_mesh(plt) -> None:
    ids, y, yq, geo, _ = train_labels()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))

    ax = axes[0, 0]
    nf = geo["n_faces"].to_numpy(float)
    ax.hist(np.log10(nf), bins=60, color="#3b6ea5")
    ax.set_xticks([0, 2, 4, 6, 8], labels=["1", "100", "10⁴", "10⁶", "10⁸"])
    med = np.nanmedian(nf)
    ax.axvline(np.log10(med), color="#b4553c", ls="--", lw=1.4, label=f"медиана {med:,.0f}".replace(",", " "))
    ax.set_xlabel("треугольников в меше (log)"); ax.set_ylabel("объектов")
    ax.set_title("Размер мешей: четыре порядка")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    nc = geo["n_components"].to_numpy(float)
    ax.hist(np.log10(np.clip(nc, 1, None)), bins=60, color="#8a5cc4")
    med = np.nanmedian(nc)
    ax.axvline(np.log10(med), color="#b4553c", ls="--", lw=1.4, label=f"медиана {med:,.0f}".replace(",", " "))
    ax.axvline(1.0, color="#4c9f70", lw=1.4, label="1 связная компонента")
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6], labels=["1", "10", "100", "10³", "10⁴", "10⁵", "10⁶"])
    ax.set_xlabel("связных компонент (log)"); ax.set_ylabel("объектов")
    ax.set_title("Связность: медиана 374 компоненты на объект")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    diag = geo["bbox_diag"].to_numpy(float)
    ax.hist(np.log10(diag), bins=60, color="#2f8f6f")
    ax.set_xlabel("диагональ bbox (log10, условные единицы)"); ax.set_ylabel("объектов")
    ax.set_title(f"Абсолютный масштаб: 1e{np.nanmin(np.log10(diag)):.0f} … 1e{np.nanmax(np.log10(diag)):.0f}")

    ax = axes[1, 1]
    cnt = np.bincount(y.sum(1))
    cnt = np.concatenate([cnt[:3], [cnt[3:].sum()]]) if len(cnt) > 3 else np.pad(cnt, (0, 4 - len(cnt)))
    bars = ax.bar(["0 (чисто)", "1", "2", "3+"], cnt, color=["#4c9f70", "#3b6ea5", "#8a5cc4", "#b4553c"])
    for bar, v in zip(bars, cnt):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v}\n{100 * v / len(y):.0f}%", ha="center", fontsize=8)
    ax.set_ylim(0, cnt.max() * 1.3)
    ax.set_ylabel("объектов"); ax.set_xlabel("дефектов на объект")
    ax.set_title("Мультилейбл, но почти всегда один дефект")
    fig.tight_layout()
    fig.savefig(OUT / "fig_data_mesh.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_data_mesh.png")


def fig_cooc(plt) -> None:
    ids, y, _, _, _ = train_labels()
    n = len(y)
    yf = y.astype(np.float32)  # int8 matmul would overflow the counts (support > 127)
    joint = (yf.T @ yf) / n
    prior = y.mean(0)
    cond = joint / prior[:, None]  # P(j | i)
    np.fill_diagonal(cond, np.nan)
    order = np.argsort(prior)[::-1]
    names = [LABELS[i] for i in order]
    C = cond[np.ix_(order, order)] * 100

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(C, cmap="magma_r", vmin=0, vmax=np.nanmax(C))
    ax.set_xticks(range(10)); ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(names)
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, f"{100 * prior[order[i]]:.0f}%", ha="center", va="center", fontsize=7, color="#333")
            else:
                ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if C[i, j] > 0.55 * np.nanmax(C) else "#222")
    ax.set_title("P(дефект j | дефект i), %   ·   на диагонали — частота класса")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cooc.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_cooc.png")


def _oof_bundle():
    """Reference OOF matrix + thresholds (fused s+b if present, else probe s)."""
    fuse = CACHE / "oof_fuse_s+b.npy"
    if fuse.exists():
        oof = np.load(fuse).astype(np.float32)
        thr = np.asarray(json.loads((CACHE / "fuse_s+b.json").read_text())["thresholds"], dtype=np.float32)
        return oof, thr, "s+b"
    oof = np.load(CACHE / "oof_probe_s.npy").astype(np.float32)
    import torch

    thr = np.asarray(torch.load(CACHE / "probe_s.pt", map_location="cpu", weights_only=False)["thresholds"], dtype=np.float32)
    return oof, thr, "s"


def fig_errors(plt) -> None:
    ids, y, yq, _, _ = train_labels()
    oof, thr, tag = _oof_bundle()
    d = (oof >= thr).astype(np.int8)
    rows = []
    for k, name in enumerate(LABELS):
        tp = int(((d[:, k] == 1) & (y[:, k] == 1)).sum())
        fp = int(((d[:, k] == 1) & (y[:, k] == 0)).sum())
        fn = int(((d[:, k] == 0) & (y[:, k] == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rows.append({"label": name, "support": int(y[:, k].sum()), "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)})
    err = pd.DataFrame(rows).sort_values("support", ascending=True)
    q_pred = (d.sum(1) == 0).astype(np.int8)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.5, 1.15, 1]})
    ax = axes[0]
    yy = np.arange(len(err))
    ax.barh(yy, err["fp"], color="#b4553c", label="FP — ложное срабатывание")
    ax.barh(yy, -err["fn"], color="#3b6ea5", label="FN — пропуск")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.label} ({r.support})" for r in err.itertuples()], fontsize=9)
    ax.set_xlabel("объектов"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Структура ошибок (OOF, {tag})")

    ax = axes[1]
    x = np.arange(len(err))
    ax.bar(x - 0.2, err["precision"], width=0.38, color="#2f8f6f", label="precision")
    ax.bar(x + 0.2, err["recall"], width=0.38, color="#3b6ea5", label="recall")
    ax.set_xticks(x); ax.set_xticklabels(err["label"], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("значение"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Точность против полноты по классам")

    ax = axes[2]
    tp = int(((yq == 1) & (q_pred == 1)).sum()); fp = int(((yq == 1) & (q_pred == 0)).sum())
    fn = int(((yq == 0) & (q_pred == 1)).sum()); tn = int(((yq == 0) & (q_pred == 0)).sum())
    vals = [tp, fp, fn, tn]
    bars = ax.bar(["TP\nчисто", "FP\nложная\nтревога", "FN\nпропуск", "TN\nдефект"], vals,
                  color=["#4c9f70", "#b4553c", "#3b6ea5", "#9aa7b1"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v}", ha="center", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"quality: F1 = {2 * tp / max(2 * tp + fp + fn, 1):.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_errors.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_errors.png")


def fig_pipeline(plt) -> None:
    """Token-flow diagram of the frozen-backbone + attentive-probe solution."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    ax.set_xlim(0, 12.8); ax.set_ylim(1.15, 4.62); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc="#eef2f7", ec="#3b6ea5", fs=8.6, weight="normal", textcolor="#16324f"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=textcolor, weight=weight, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#3b6ea5", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                     color=color, lw=1.2, shrinkA=1, shrinkB=1))

    # image branch
    box(0.05, 3.30, 1.75, 0.95, "PNG-коллаж\n1536×1024\n6 рендеров 512²")
    box(2.05, 3.30, 2.05, 0.95, "DINOv3 ViT-S/B\nзаморожен\n32×32 патчей × D", fc="#e8f0f8")
    box(4.35, 3.30, 1.95, 0.95, "пулинг патчей\n→ сетка + mean/max", fc="#e8f0f8")
    arrow(1.82, 3.78, 2.03, 3.78); arrow(4.12, 3.78, 4.33, 3.78)

    # mesh branch
    box(0.05, 1.90, 1.75, 0.95, ".npz меш\nвершины + грани\n≤ 2.6·10⁷ граней")
    box(2.05, 1.90, 2.05, 0.95, "geometry.py\n27 признаков\nразмер, топология")
    box(4.35, 1.90, 1.95, 0.95, "z-score нормализация\n→ MLP(27→384)", fc="#eef7ee", ec="#2f8f6f", textcolor="#1c4a39")
    arrow(1.82, 2.38, 2.03, 2.38, color="#2f8f6f"); arrow(4.12, 2.38, 4.33, 2.38, color="#2f8f6f")

    # token sequence
    box(6.65, 2.05, 2.55, 2.15,
        "токены × 384\n\n6×[mean] + 6×[max]\n6×[grid-сетка]\n+ [GEOM] + [query]",
        fc="#f7f3fb", ec="#8a5cc4", textcolor="#3d2a5c", fs=8.4)
    arrow(6.32, 3.78, 6.63, 3.55, color="#3b6ea5")
    arrow(6.32, 2.38, 6.63, 2.70, color="#2f8f6f")

    box(9.55, 3.25, 2.05, 1.05, "Transformer\n4 слоя, d=384\n7.73M обучаемых", fc="#fdf3e7", ec="#b4553c", textcolor="#5c2c1c")
    arrow(9.22, 3.28, 9.53, 3.45, color="#8a5cc4")

    # head
    box(9.55, 1.90, 2.05, 0.95, "LayerNorm → Linear\n384 → 10\nsigmoid ×10", fc="#fdf3e7", ec="#b4553c", textcolor="#5c2c1c")
    arrow(10.57, 3.23, 10.57, 2.87, color="#b4553c")
    box(11.85, 1.90, 0.9, 0.95, "пороги\n→ csv", fc="#f2f5f8", ec="#1f3b57", fs=8.0)
    arrow(11.62, 2.38, 11.83, 2.38, color="#b4553c")

    ax.text(6.4, 1.35, "quality = «ни одного дефекта выше порога» — правило разметки подтверждено на всех 8964 объектах",
            fontsize=9, color="#555", style="italic")
    ax.text(6.4, 4.45, "Замороженные признаки (обучается только проб) + дешёвая обучаемая агрегация",
            fontsize=10.5, weight="bold", color="#1f3b57")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_pipeline.png")


def fig_curves(plt) -> None:
    path = ABL / "curves.json"
    if not path.exists():
        print("[figures] fig_curves skipped (cache/ablations/curves.json missing)")
        return
    data = json.loads(path.read_text())
    train_loss, val_loss = data["train_loss"], data["val_loss"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(1, len(train_loss) + 1)
    ax.plot(x, train_loss, color="#3b6ea5", lw=2, label="train (BCE с pos_weight)")
    ax.plot(x, val_loss[: len(x)], color="#b4553c", lw=2, label="val (та же потеря, OOF-фолд)")
    ax.set_xlabel("эпоха"); ax.set_ylabel("loss"); ax.set_yscale("log")
    best = int(np.argmin(val_loss)) + 1
    ax.axvline(best, color="#2f8f6f", ls="--", lw=1.2)
    ax.annotate(f"минимум val: эпоха {best}", xy=(best, min(val_loss)), xytext=(best + 1.5, min(val_loss) * 0.7),
                fontsize=9, color="#2f8f6f", arrowprops=dict(arrowstyle="->", color="#2f8f6f"))
    ax.set_title("Кривые обучения одного фолда: 40 эпох, OneCycleLR")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curves.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_curves.png")


def fig_hparams(plt) -> None:
    runs = {
        "baseline (depth 4)": None,
        "depth 2": "depth2", "depth 6": "depth6",
        "10 эпох": "epochs10",
        "pos_weight^0.25": "posw025", "pos_weight^0.75": "posw075",
        "seed 1": "seed1", "seed 2": "seed2",
    }
    names, scores, per = [], [], {}
    base_oof = CACHE / "oof_probe_s.npy"
    ids, y, yq, _, _ = train_labels()
    from mesh_quality import metric

    if base_oof.exists():
        _, res = metric.tune_thresholds(np.load(base_oof).astype(np.float32), y, yq, verbose=False)
        names.append("baseline (depth 4)"); scores.append(res["score"]); per["baseline"] = res["per_label"]
    for name, tag in runs.items():
        if tag is None or not (ABL / f"{tag}.json").exists():
            continue
        data = json.loads((ABL / f"{tag}.json").read_text())
        names.append(name); scores.append(data["score"]); per[tag] = data["per_label"]
    if len(names) < 2:
        print("[figures] fig_hparams skipped (no ablation results yet)")
        return

    order = np.argsort(scores)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    colors = ["#3b6ea5" if n == "baseline (depth 4)" else "#9aa7b1" for n in names]
    bars = ax.bar([names[i] for i in order], [scores[i] for i in order], color=[colors[i] for i in order])
    for bar, i in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, scores[i] + 0.08, f"{scores[i]:.2f}", ha="center", fontsize=9)
    ax.set_ylim(min(scores) - 0.6, max(scores) + 0.75)
    ax.set_ylabel("OOF, баллов из 20"); ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.set_title("Абляции: 5 фолдов, один и тот же сплит")
    # noise floor: same recipe, different init seed
    if "seed1" in per:
        base_score = scores[names.index("baseline (depth 4)")]
        seed_idx = next((i for i, n in enumerate(names) if n.startswith("seed")), None)
        delta = abs(base_score - scores[seed_idx]) if seed_idx is not None else 0.0
        ax.axhspan(base_score - delta, base_score + delta, color="#c9c9c9", alpha=0.55, zorder=0)
        ax.text(0.02, 0.06, f"полоса шума инициализации: ±{delta:.2f}", transform=ax.transAxes,
                fontsize=8, color="#555555")

    ax = axes[1]
    show = [n for n in ["baseline", "depth2", "epochs10", "posw025", "seed1"] if n in per]
    x = np.arange(10)
    width = 0.8 / len(show)
    palette = ["#3b6ea5", "#8a5cc4", "#c08a2e", "#2f8f6f", "#b4553c"]
    base = per["baseline"]
    for j, key in enumerate(show):
        vals = [per[key][n] - base[n] for n in LABELS] if key != "baseline" else [0] * 10
        ax.bar(x + (j - (len(show) - 1) / 2) * width, vals, width=width * 0.9,
               label=key, color=palette[j % len(palette)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ΔF1 к baseline")
    ax.set_title("Куда уходит разница")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hparams.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_hparams.png")


# --------------------------------------------------------------------------- #
# GPU figures: per-item analysis of the probe
# --------------------------------------------------------------------------- #


class ItemSession:
    """Loads one item's probe inputs once: full 32x32 tokens, tiles, geometry."""

    def __init__(self, model_key: str = "s", device: str | None = None):
        import torch

        from mesh_quality import model, visualize

        self.torch = torch
        self.model = model
        self.visualize = visualize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CACHE / f"probe_{model_key}.pt", map_location=self.device, weights_only=False)
        self.cfg = model.ProbeConfig(**ckpt["cfg"])
        self.net = model.AttentiveProbe(self.cfg).to(self.device).eval()
        self.net.load_state_dict(ckpt["states"][0])
        self.ckpt = ckpt
        self.thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)

    def item(self, item_id: str, split: str) -> dict:
        torch = self.torch
        model, visualize = self.model, self.visualize
        from mesh_quality import images

        tokens, tiles = visualize._full_grid_features(TASK / "data" / split / f"{item_id}.png",
                                                      self.device, self.ckpt["model_key"])
        dim = tokens.shape[-1]
        grid = images.pool_patches(
            torch.from_numpy(tokens.reshape(6, images.GRID, images.GRID, dim))
        ).numpy()
        geom, _ = model._geometry_table(CACHE, split, np.asarray([item_id]))
        geom = model.standardise_geom(geom, self.ckpt["geom_median"], self.ckpt["geom_scale"]).astype(np.float32)
        t = torch.tensor
        return {
            "id": item_id, "split": split, "tiles": tiles,
            "mean": t(tokens.mean(1)[None].astype(np.float32), device=self.device),
            "max": t(tokens.max(1)[None].astype(np.float32), device=self.device),
            "grid": t(grid[None].astype(np.float32), device=self.device),
            "geom": t(geom, device=self.device),
        }

    def logits(self, it: dict, **overrides):
        mean = overrides.get("mean", it["mean"]); mx = overrides.get("max", it["max"])
        grid = overrides.get("grid", it["grid"]); geom = overrides.get("geom", it["geom"])
        return self.net(mean, mx, grid, geom)[0]

    def probs(self, it: dict) -> np.ndarray:
        with self.torch.inference_mode():
            return self.torch.sigmoid(self.logits(it)).cpu().numpy()

    def saliency(self, it: dict, label: str) -> np.ndarray:
        grid = it["grid"].clone().requires_grad_(True)
        logits = self.logits(it, grid=grid)
        k = LABELS.index(label)
        self.net.zero_grad(set_to_none=True)
        logits[k].backward()
        sal = (grid.grad * grid).abs().sum(-1)[0].detach().cpu().numpy()
        return sal / (sal.max() + 1e-9)

    def occlusion(self, it: dict, label: str) -> np.ndarray:
        """Δlogit when one 8x8 cell is zeroed, over all 6 tiles."""
        torch = self.torch
        k = LABELS.index(label)
        with torch.inference_mode():
            base = float(self.logits(it)[k])
            out = np.zeros((6, self.cfg.grid, self.cfg.grid), dtype=np.float32)
            for t in range(6):
                for r in range(self.cfg.grid):
                    for c in range(self.cfg.grid):
                        g = it["grid"].clone()
                        g[0, t, r, c] = 0.0
                        out[t, r, c] = base - float(self.logits(it, grid=g)[k])
        return np.clip(out, 0, None) / (out.max() + 1e-9)

    def attention(self, it: dict) -> np.ndarray:
        """Query-token attention to grid cells, averaged over layers and heads."""
        torch = self.torch
        net = self.net
        with torch.inference_mode():
            x = net.tokens(it["mean"], it["max"], it["grid"], it["geom"])
            attn = []
            for layer in net.blocks.layers:
                h = layer.norm1(x)
                a, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
                x = x + layer.dropout1(a)
                h = layer.norm2(x)
                h = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
                x = x + layer.dropout2(h)
                attn.append(w.detach())
            attn = torch.stack(attn)                     # [L, 1, H, T, T]
            q = attn[:, 0, :, -1, :].mean(dim=(0, 1))    # attention from the query token
            # token layout: [6 tile-mean | 6 tile-max | 6*g*g grid | GEOM | query]
            cfg = self.cfg
            n_prefix = 12  # the 12 tile-summary tokens precede the grid block
            grid = q[n_prefix: n_prefix + 6 * cfg.grid * cfg.grid]
            grid = grid.reshape(6, cfg.grid, cfg.grid).cpu().numpy()
        return (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)

    def view_ablation(self, it: dict) -> np.ndarray:
        """Δlogit per label when an entire view is removed (all its tokens zeroed)."""
        torch = self.torch
        with torch.inference_mode():
            base = self.logits(it).cpu().numpy()
            delta = np.zeros((6, 10), dtype=np.float32)
            for t in range(6):
                mean, mx, grid = it["mean"].clone(), it["max"].clone(), it["grid"].clone()
                mean[0, t] = 0.0; mx[0, t] = 0.0; grid[0, t] = 0.0
                delta[t] = base - self.logits(it, mean=mean, max=mx, grid=grid).cpu().numpy()
        return delta


def _sheet(session: ItemSession, it: dict, heat: np.ndarray) -> np.ndarray:
    return session.visualize._contact_sheet(it["tiles"], heat)


def _sheet_views(session: ItemSession, it: dict, heat: np.ndarray,
                 views: tuple[int, ...] = (0, 1, 2), tile_px: int = 260) -> np.ndarray:
    """Contact sheet over a subset of views (one row) — bigger renders than _contact_sheet."""
    from PIL import Image

    from mesh_quality import visualize as V

    heats = np.asarray(heat)
    floor = float(np.quantile(heats.reshape(-1), 0.94))
    row = []
    for k in views:
        tile = np.asarray(Image.fromarray(it["tiles"][k]).resize((tile_px, tile_px), Image.BILINEAR),
                          dtype=np.uint8)
        row.append(V._overlay(tile, heats[k], floor=floor, alpha=0.9))
    return np.concatenate(row, axis=1)


def fig_methods(plt, session: ItemSession) -> None:
    item_id = "f08554bd-68a7-43d6-af5a-e0f8bd43e093"  # test item, confident `artifacts`
    item = session.item(item_id, "test")
    probs = session.probs(item)
    k = LABELS.index("artifacts")
    print(f"[figures] fig_methods: {item_id[:8]} artifacts p={probs[k]:.3f}")
    sheets = {
        "градиент × активация": session.saliency(item, "artifacts"),
        "окклюзия ячейки (Δlogit)": session.occlusion(item, "artifacts"),
        "внимание query → ячейки": session.attention(item),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 2.9))
    for ax, (title, heat) in zip(axes, sheets.items()):
        ax.imshow(_sheet_views(session, item, heat)); ax.axis("off")
        ax.set_title(title, fontsize=10.5)
    fig.suptitle(f"Три метода объяснения одной и той же улики: `artifacts`, p = {probs[k]:.2f}  ·  {item_id[:8]}"
                 f"   (виды az0, az90, az180; цвет ячейки = вклад)",
                 y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_methods.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_methods.png")


def fig_views(plt, session: ItemSession, cases: list[tuple[str, str]]) -> None:
    rows = []
    for item_id, split in cases:
        it = session.item(item_id, split)
        probs = session.probs(it)
        delta = session.view_ablation(it)
        rows.append((item_id, split, probs, delta))

    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 3.9), squeeze=False)
    vmax = float(np.quantile(np.abs(np.stack([d for *_, d in rows])), 0.98))
    for ax, (item_id, split, probs, delta) in zip(axes[0], rows):
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(6)); ax.set_yticklabels(VIEWS, fontsize=9)
        ax.set_xticks(range(10)); ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{item_id[:8]} ({split})\n" + " ".join(f"{n}={probs[LABELS.index(n)]:.2f}"
                     for n in np.asarray(LABELS)[probs > 0.5]), fontsize=9)
        ax.grid(False)
        for t in range(6):
            for j in range(10):
                if abs(delta[t, j]) > 0.3 * vmax:
                    ax.text(j, t, f"{delta[t, j]:+.1f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.06,
                 pad=0.22, shrink=0.5, label="Δlogit при удалении вида")
    fig.suptitle("Вклад каждого рендера: удаляем один вид, смотрим, как падает логит", y=1.04, fontsize=11)
    fig.savefig(OUT / "fig_views.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_views.png")


def fig_cases(plt, session: ItemSession) -> None:
    """Failure cases chosen from the OOF matrix + the two confident test examples."""
    ids, y, yq, _, _ = train_labels()
    oof, thr, tag = _oof_bundle()
    d = (oof >= thr).astype(np.int8)
    picks: list[tuple[str, str, str, str]] = []  # id, split, label, reason

    for label, want_true, want_pred, reason, clean_only in (
        ("artifacts", 1, 0, "FN: `artifacts` есть, модель промолчала", False),
        ("abstract", 0, 1, "FP: чистый объект назван `abstract`", True),
    ):
        k = LABELS.index(label)
        cand = np.where((y[:, k] == want_true) & (d[:, k] == want_pred) & ((yq == 1) if clean_only else True))[0]
        score = oof[cand, k] if want_pred == 1 else -oof[cand, k]
        idx = int(cand[int(np.argmax(score))])
        picks.append((str(ids[idx]), "train", label, reason))

    probs_test = np.load(CACHE / "probs_test_fuse_s+b.npy").astype(np.float32)
    test_ids = np.asarray(pd.read_csv(TASK / "data" / "test.csv")["item_id"].astype(str))
    for label in ("artifacts", "open"):
        k = LABELS.index(label)
        best = int(np.argmax(probs_test[:, k]))
        picks.append((str(test_ids[best]), "test", label, f"тест: уверенный `{label}` p={probs_test[best, k]:.2f}"))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2))
    for ax, (item_id, split, label, reason) in zip(axes.ravel(), picks):
        it = session.item(item_id, split)
        probs = session.probs(it)
        k = LABELS.index(label)
        heat = session.saliency(it, label)
        ax.imshow(_sheet_views(session, it, heat)); ax.axis("off")
        shown = ", ".join(f"{LABELS[j]} {probs[j]:.2f}" for j in np.argsort(probs)[::-1][:3])
        if split == "train":
            row = int(np.where(ids == item_id)[0][0])
            truth = "истина: " + (", ".join(LABELS[j] for j in np.where(y[row] > 0)[0]) or "чисто")
        else:
            truth = "тест: разметки нет, показана только уверенность"
        ax.set_title(f"{reason}\n{item_id[:8]} · p({label}) = {probs[k]:.2f} · {truth}", fontsize=8.5, loc="left")
        print(f"[figures] case {reason} -> {item_id} top: {shown}")
    fig.suptitle("Успехи и ошибки: где модель «смотрит» (виды az0/az90/az180, градиентная атрибуция, проб s)", y=1.0, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cases.png", bbox_inches="tight"); plt.close(fig)
    print("[figures] fig_cases.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-only", action="store_true", help="skip the per-item GPU figures")
    ap.add_argument("--only", default=None, help="comma-separated figure names to build")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    wanted = set(args.only.split(",")) if args.only else None

    def go(name: str) -> bool:
        return wanted is None or name in wanted

    if go("data"): fig_data_mesh(plt)
    if go("cooc"): fig_cooc(plt)
    if go("errors"): fig_errors(plt)
    if go("pipeline"): fig_pipeline(plt)
    if go("curves"): fig_curves(plt)
    if go("hparams"): fig_hparams(plt)

    if args.cpu_only or (wanted is not None and not (wanted & {"methods", "views", "cases"})):
        return 0

    import torch

    if not torch.cuda.is_available():
        print("[figures] no CUDA -- skipping the per-item figures")
        return 0
    session = ItemSession()
    if go("methods"):
        fig_methods(plt, session)
    if go("views"):
        fig_views(plt, session, [
            ("f08554bd-68a7-43d6-af5a-e0f8bd43e093", "test"),  # confident artifacts (no labels)
            ("f4e78e59-9495-416b-be05-21f62e2638b3", "test"),   # confident open (no labels)
        ])
    if go("cases"):
        fig_cases(plt, session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
