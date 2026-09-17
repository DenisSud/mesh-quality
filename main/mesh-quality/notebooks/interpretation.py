# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Контроль качества 3D-моделей — разбор решения и ошибок
#
# Второй ноутбук решения: **интерпретация**. Он показывает, что именно видит модель,
# на чём ошибается и почему, и какие выводы из этого следуют.
#
# Содержание:
#
# 1. данные и метрика: дисбаланс классов, тождество `quality ⇔ нет дефектов`,
#    вклад каждого класса в 20 баллов;
# 2. признаки: что даёт геометрия и что добавляют изображения (по классам);
# 3. разбор ошибок на out-of-fold предсказаниях: FP/FN по классам, уверенность,
#    типичные промахи;
# 4. карты внимания по патчам: 4 разобранных случая (успехи и ошибки);
# 5. тестовая часть: распределение предсказаний, совместная встречаемость
#    дефектов, самые неуверенные объекты;
# 6. выводы и что улучшать.
#
# Ноутбук не требует сырых данных: рендеры разобранных случаев и все матрицы
# вероятностей лежат в бандле решения (`cases.csv`, `cache/*.npy`).

# %% [markdown]
# ## 0. Окружение и данные

# %% tags=["skip"]
# !pip -q install "timm==1.0.29" gdown

# %%
import hashlib
import json
import random
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

SEED = 42
random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

np.random.seed(SEED)
torch.manual_seed(SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device: {DEVICE} | seed {SEED}")

# %%
WORK = Path("/content/mesh_quality_work") if Path("/content").exists() else Path.cwd() / "mesh_quality_work"
WORK.mkdir(parents=True, exist_ok=True)

BUNDLE_URL = ""  # тот же бандл, что и в colab_solution.ipynb
BUNDLE_GDRIVE_ID = ""


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size:
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dst.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    print(f"скачано: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def fetch(source: str, dst: Path) -> Path:
    if "disk.360.yandex.ru" in source or "yadi.sk" in source:
        api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        with urllib.request.urlopen(f"{api}?public_key={urllib.parse.quote(source, safe='')}") as r:
            source = json.loads(r.read())["href"]
    if "://" in source:
        return _download(source, dst)
    import gdown

    gdown.download(id=source, output=str(dst), quiet=False, fuzzy=True)
    return dst


bundle_zip = fetch(BUNDLE_URL or BUNDLE_GDRIVE_ID, WORK / "mesh_quality_bundle.zip")
with zipfile.ZipFile(bundle_zip) as zf:
    zf.extractall(WORK)
TASK = WORK / "mesh_quality"
CACHE, DATA = TASK / "cache", TASK / "data"
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import metric, model, visualize  # noqa: E402

manifest = json.loads((TASK / "manifest.json").read_text())
ref = manifest["reference"]
thresholds = np.asarray(ref["thresholds"], dtype=np.float32)
print(f"эталон: {ref['model']} (модели {ref['models']}, слияние: {ref['fused']}), "
      f"OOF {ref['cv']['score']:.3f} / 20")

labels_df = pd.read_csv(DATA / "train.csv").set_index("item_id")
train_ids = np.asarray(pd.read_csv(DATA / "train.csv")["item_id"].astype(str))
y = labels_df.loc[list(train_ids), list(metric.DEFECTS)].to_numpy(dtype=np.int8)
yq = metric.derive_quality(y)
oof = np.load(TASK / ref["reference_oof"]).astype(np.float32)
d_pred = (oof >= thresholds).astype(np.int8)
q_pred = metric.derive_quality(d_pred)
print(f"OOF: {len(y)} объектов, предсказано «чистых» {int(q_pred.sum())}, истинных {int(yq.sum())}")

# %%
cases = pd.read_csv(TASK / "cases.csv")
print(f"разобранных случаев в бандле: {len(cases)}")
cases[["item_id", "split", "reason"]].assign(item_id=lambda d: d.item_id.str[:8])

# %% [markdown]
# ## 1. Данные и метрика
#
# Метрика соревнования: `10 · F1(quality) + 10 · F1_weighted(defects)`.
# Веса `F1_weighted` — доли классов, поэтому бюджет баллов распределён очень
# неравномерно: `noisy` (28% объектов) «стоит» 2.75 балла, а `intersection`
# и `scale` (по 1.2%) — по 0.12.

# %%
support = y.sum(0)
total = support.sum()
share = support / total
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
order = np.argsort(support)[::-1]
names = [metric.DEFECTS[i] for i in order]
axes[0].bar(range(10), support[order], color="#3b6ea5")
axes[0].set_xticks(range(10))
axes[0].set_xticklabels(names, rotation=45, ha="right")
axes[0].set_title(f"Объектов с дефектом (n={len(y)})")
for i, v in enumerate(support[order]):
    axes[0].text(i, v + 40, f"{v}", ha="center", fontsize=7)

axes[1].bar(range(10), 10 * share[order], color="#b4553c")
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(names, rotation=45, ha="right")
axes[1].set_title("Бюджет баллов: 10 · вес класса")
for i, v in enumerate(10 * share[order]):
    axes[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

identical = bool((metric.derive_quality(y) == labels_df.loc[list(train_ids), "quality"].to_numpy(np.int8)).all())
axes[2].bar(["quality=1", "quality=0"], [int(yq.sum()), len(y) - int(yq.sum())], color=["#4c9f70", "#7a7a7a"])
axes[2].set_title(f"Мишень quality\n(«нет дефектов» ⇔ quality=1: {identical})")
for i, v in enumerate([int(yq.sum()), len(y) - int(yq.sum())]):
    axes[2].text(i, v + 60, f"{v} ({100 * v / len(y):.0f}%)", ha="center", fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# Поскольку `quality` выводится из дефектов, а не предсказывается отдельно,
# единственный способ улучшить первую половину счёта — уменьшить число ложных
# срабатываний дефектов. Это же делает `quality` идеальным «детектором» общей
# точности: 15% объектов чистые, и один лишний дефект уносит объект из этого класса.

# %%
grid = np.linspace(0.02, 0.98, 97)


def f1_curve(k: int) -> np.ndarray:
    out = []
    for t in grid:
        pred = (oof[:, k] >= t).astype(np.int8)
        tp = int(((pred == 1) & (y[:, k] == 1)).sum())
        fp = int(((pred == 1) & (y[:, k] == 0)).sum())
        fn = int(((pred == 0) & (y[:, k] == 1)).sum())
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.asarray(out)


show = ["noisy", "abstract", "artifacts", "intersection"]
fig, axes = plt.subplots(1, len(show), figsize=(13.5, 3.0), sharey=True)
for ax, name in zip(axes, show):
    k = metric.DEFECTS.index(name)
    curve = f1_curve(k)
    best = int(curve.argmax())
    ax.plot(grid, curve, color="#3b6ea5", lw=2)
    ax.axvline(thresholds[k], color="#b4553c", ls="--", lw=1.1)
    ax.plot(grid[best], curve[best], "o", color="#b4553c", ms=4)
    ax.set_title(f"{name}: F1_max {curve.max():.2f} @ {grid[best]:.2f}\nвыбран порог {thresholds[k]:.2f}")
    ax.set_xlabel("порог")
axes[0].set_ylabel("F1")
fig.suptitle("F1 класса как функция порога (OOF): плато широкое, но у редких классов оптимум смещён вправо", y=1.06)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Что добавляют изображения
#
# Сравним два источника сигнала по классам: 27 геометрических признаков
# (HistGradientBoosting) и слияние DINOv3-пробов (ViT-S + ViT-B). Первый видит структуру меша, второй —
# то, как объект выглядит.

# %%
from sklearn.metrics import f1_score  # noqa: E402

oof_geometry = np.load(CACHE / "oof_geometry.npy").astype(np.float32) if (CACHE / "oof_geometry.npy").exists() else None
per_label = {"DINOv3 + проб (слияние)": metric.tune_thresholds(oof, y, yq, verbose=False)[1]["per_label"]}
if oof_geometry is not None:
    per_label["геометрия (HistGB)"] = metric.tune_thresholds(oof_geometry, y, yq, verbose=False)[1]["per_label"]

order = np.argsort([per_label["DINOv3 + проб (слияние)"][n] for n in metric.DEFECTS])
fig, ax = plt.subplots(figsize=(11.5, 3.2))
width = 0.8 / len(per_label)
for j, (tag, values) in enumerate(per_label.items()):
    ax.bar(np.arange(10) + (j - (len(per_label) - 1) / 2) * width,
           [values[metric.DEFECTS[i]] for i in order], width=width * 0.9, label=tag,
           color=["#3b6ea5", "#9aa7b1"][j % 2])
ax.set_xticks(range(10))
ax.set_xticklabels([metric.DEFECTS[i] for i in order], rotation=35, ha="right")
ax.set_ylabel("F1 (OOF)")
ax.set_title("По классам: какие дефекты ловятся взглядом, а какие — геометрией")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
plt.show()

best_image = sorted(metric.DEFECTS, key=lambda n: per_label["DINOv3 + проб (слияние)"][n] - (per_label.get("геометрия (HistGB)", {}).get(n, 0)))[::-1][:3]
worst = sorted(metric.DEFECTS, key=lambda n: per_label["DINOv3 + проб (слияние)"][n])[:4]
print("изображения дают больше всего:", ", ".join(best_image))
print("остаются самыми слабыми:", ", ".join(f"{n} ({per_label['DINOv3 + проб (слияние)'][n]:.2f})" for n in worst))

# %% [markdown]
# ### Какие геометрические признаки вообще информативны
#
# Взаимная информация каждого из 27 признаков с каждым дефектом: строки —
# признаки, столбцы — классы. Видно, что `noisy` и `lowpoly` предсказуемы
# из геометрии, а `artifacts`/`open`/`intersection` — почти нет: это локальные
# дефекты, которые нужно *увидеть*.

# %%
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

geom, geom_names = model._geometry_table(CACHE, "train", train_ids)
geom = np.nan_to_num(np.asarray(geom, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
assert geom.shape[0] == len(y), geom.shape
assert len(geom_names) == geom.shape[1], (len(geom_names), geom.shape)

mi = np.zeros((geom.shape[1], len(metric.DEFECTS)))
for j in range(len(metric.DEFECTS)):
    mi[:, j] = mutual_info_classif(geom, y[:, j], random_state=SEED)

fig, ax = plt.subplots(figsize=(11.5, 5.2))
im = ax.imshow(mi, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(metric.DEFECTS)))
ax.set_xticklabels(metric.DEFECTS, rotation=35, ha="right")
ax.set_yticks(range(mi.shape[0]))
ax.set_yticklabels([n.replace("geom_", "") for n in geom_names], fontsize=7)
ax.set_title("Взаимная информация признак × дефект (27 признаков, train)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
plt.show()

top = pd.DataFrame(mi, index=list(geom_names), columns=list(metric.DEFECTS))
print("топ-3 признака для каждого класса:")
for name in worst[:4]:
    print(f"  {name:12s} " + ", ".join(f"{i} ({v:.3f})" for i, v in top[name].sort_values(ascending=False).head(3).items()))

# %% [markdown]
# ## 3. Разбор ошибок на out-of-fold предсказаниях
#
# Три взгляда на ошибки: (а) сколько FP/FN у каждого класса, (б) как уверенность
# модели связана с правильностью, (в) как устроены ошибки флага `quality`.

# %%
rows = []
for k, name in enumerate(metric.DEFECTS):
    tp = int(((d_pred[:, k] == 1) & (y[:, k] == 1)).sum())
    fp = int(((d_pred[:, k] == 1) & (y[:, k] == 0)).sum())
    fn = int(((d_pred[:, k] == 0) & (y[:, k] == 1)).sum())
    tn = int(((d_pred[:, k] == 0) & (y[:, k] == 0)).sum())
    rows.append({
        "класс": name, "support": int(support[k]), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
        "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
    })
errors = pd.DataFrame(rows).set_index("класс")
errors["FP/FN"] = (errors.FP / errors.FN.clip(lower=1)).round(2)
try:
    display(errors)  # noqa: F821
except NameError:
    print(errors)

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
axes[0].barh(errors.index[::-1], errors.FP[::-1], color="#b4553c", label="FP (ложные)")
axes[0].barh(errors.index[::-1], -errors.FN[::-1], color="#3b6ea5", label="FN (пропуски)")
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_title("Структура ошибок по классам")
axes[0].legend(frameon=False, fontsize=8)

for k, name in enumerate(["noisy", "artifacts"]):
    j = metric.DEFECTS.index(name)
    ax = axes[1 + k]
    correct = (d_pred[:, j] == y[:, j])
    ax.hist(oof[correct, j], bins=30, alpha=0.75, label="верно", color="#4c9f70")
    ax.hist(oof[~correct, j], bins=30, alpha=0.75, label="ошибка", color="#b4553c")
    ax.axvline(thresholds[j], color="black", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"{name}: уверенность и ошибки")
    ax.set_xlabel("предсказанная вероятность")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# %%
quality_rows = {
    "истинно чистые (quality=1)": int(yq.sum()),
    "предсказано чистых": int(q_pred.sum()),
    "верно чистых (TP)": int(((yq == 1) & (q_pred == 1)).sum()),
    "чистый объект назван дефектным (FP)": int(((yq == 1) & (q_pred == 0)).sum()),
    "дефектный назван чистым (FN)": int(((yq == 0) & (q_pred == 1)).sum()),
}
for k, v in quality_rows.items():
    print(f"{k:38s} {v}")
print(f"\nF1(quality) = {metric.tune_thresholds(oof, y, yq, verbose=False)[1]['quality_f1']:.4f} "
      f"-> {10 * metric.tune_thresholds(oof, y, yq, verbose=False)[1]['quality_f1']:.2f} баллов из 10")

pred_defects = d_pred.sum(1)
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.hist(pred_defects[yq == 1], bins=range(0, 8), alpha=0.8, label="истинно чистые", color="#4c9f70")
ax.hist(pred_defects[yq == 0], bins=range(0, 8), alpha=0.5, label="дефектные", color="#b4553c")
ax.set_yscale("log")
ax.set_xlabel("сколько дефектов предсказано")
ax.set_ylabel("объектов (log)")
ax.set_title("Один лишний дефект переводит объект в «дефектные»")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Карты внимания: разобранные случаи
#
# Атрибуция считается по полной сетке `32×32` патчей: градиент логита класса по
# входным признакам ячеек, `|grad × activation|`. Оранжево-красные ячейки —
# области, на которые модель опирается сильнее всего; подсвечивается top-15%
# ячеек, остальное остаётся исходным рендером. Показаны 8 обучающих случаев
# (успехи и ошибки с известной истиной) и 2 тестовых объекта с самыми уверенными
# предсказаниями `artifacts` и `open`.

# %%
out_dir = WORK / "attribution"
out_dir.mkdir(exist_ok=True)


def explain_case(row) -> dict:
    labels_arg = row.pred_defects.replace(" ", ",") if isinstance(row.pred_defects, str) else ""
    args = type("Args", (), {})()
    args.task_dir = TASK
    args.data_dir = DATA
    # Атрибуция считается для одного проба: берём первую модель слияния (s).
    args.model = ref["models"][0]
    args.split = row.split
    args.item_id = row.item_id
    args.labels = labels_arg or None
    args.device = DEVICE
    args.out_dir = out_dir
    visualize.run(args)
    return json.loads((out_dir / f"{row.item_id}.json").read_text())


explained = {}
for _, row in cases.iterrows():
    explained[row.item_id] = explain_case(row)
print(f"готово карт: {len(explained)} (включая тестовые объекты: у них нет разметки, "
      f"но есть рендеры и предсказания)")

# %%
from IPython.display import Image, Markdown, display  # noqa: E402

for _, row in cases.iterrows():
    summary = explained.get(row.item_id)
    if summary is None:
        continue
    truth = (f"истина: `{row.true_defects or '—'}` (quality={row.true_quality})" if row.split == "train"
             else "истина: нет разметки (тестовая часть)")
    display(Markdown(
        f"**{row.reason}** — `{row.item_id[:8]}` ({row.split})\n\n"
        f"- {truth}\n"
        f"- предсказано: `{row.pred_defects or '—'}`\n"
    ))
    for name, payload in summary["labels"].items():
        top = payload["top_cells"][0]
        display(Markdown(
            f"`{name}`: p = {payload['probability']:.3f} (порог {payload['threshold']:.2f}); "
            f"главная ячейка: тайл {top['tile']} ({['az0','az90','az180','az270','сверху','снизу'][top['tile']]}), "
            f"строка {top['row']}, столбец {top['col']}"
        ))
        display(Image(filename=str(out_dir / f"{row.item_id}_{name}.png"), width=760))

# %% [markdown]
# ## 5. Тестовая часть
#
# Для тестовых объектов разметки нет, поэтому смотрим на структуру
# предсказаний: сколько дефектов модель ставит, насколько распределение
# отличается от обучающего, какие объекты самые неуверенные, и как дефекты
# встречаются вместе. Именно эта картина показывает, где решение «осторожно»,
# а где — рискованно.

# %%
probs_test = np.load(TASK / ref["reference_probs"]).astype(np.float32)
test_ids = np.asarray(pd.read_csv(DATA / "test.csv")["item_id"].astype(str))
d_test = (probs_test >= thresholds).astype(np.int8)
q_test = metric.derive_quality(d_test)
print(f"тестовых объектов: {len(test_ids)}; предсказано «чистых»: {int(q_test.sum())} "
      f"({100 * q_test.mean():.1f}% против {100 * yq.mean():.1f}% в train)")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
x = np.arange(10)
axes[0].bar(x - 0.2, 100 * support / len(y), width=0.38, label="train (истина)", color="#9aa7b1")
axes[0].bar(x + 0.2, 100 * d_test.sum(0) / len(test_ids), width=0.38, label="test (предсказано)", color="#3b6ea5")
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric.DEFECTS, rotation=40, ha="right")
axes[0].set_ylabel("% объектов")
axes[0].set_title("Частота дефектов: train vs предсказания на тесте")
axes[0].legend(frameon=False, fontsize=8)

axes[1].hist(d_test.sum(1), bins=range(0, 9), color="#3b6ea5")
axes[1].set_title(f"Сколько дефектов у объекта\n(среднее {d_test.sum(1).mean():.2f}, train {y.sum(1).mean():.2f})")
axes[1].set_xlabel("число предсказанных дефектов")

co = d_test.T @ d_test
np.fill_diagonal(co, 0)
axes[2].imshow(co, cmap="magma")
axes[2].set_xticks(x)
axes[2].set_xticklabels(metric.DEFECTS, rotation=40, ha="right", fontsize=7)
axes[2].set_yticks(x)
axes[2].set_yticklabels(metric.DEFECTS, fontsize=7)
axes[2].set_title("Совместная встречаемость (test)")
fig.tight_layout()
plt.show()

# %%
margin = np.abs(probs_test - thresholds).min(1)  # насколько объект «близок» к смене решения
uncertain = np.argsort(margin)[:15]
flip_thr = thresholds + np.sign(thresholds - probs_test) * margin[:, None]
near = (np.abs(probs_test - thresholds) < 0.05).sum(0)
print("ячеек решения в пределах ±0.05 от порога:", dict(zip(metric.DEFECTS, near.tolist())))
table = pd.DataFrame({
    "item_id": [i[:8] for i in test_ids[uncertain]],
    "самый близкий класс": [metric.DEFECTS[int(np.argmin(np.abs(probs_test[i] - thresholds)))] for i in uncertain],
    "p": [round(float(probs_test[i][int(np.argmin(np.abs(probs_test[i] - thresholds)))]), 3) for i in uncertain],
    "порог": [round(float(thresholds[int(np.argmin(np.abs(probs_test[i] - thresholds)))]), 2) for i in uncertain],
    "предсказано дефектов": d_test[uncertain].sum(1),
})
try:
    display(table)  # noqa: F821
except NameError:
    print(table)

# %% [markdown]
# ## 6. Выводы
#
# * **Метрика — про осторожность.** Бюджет баллов смещён к частым классам, но
#   `quality` наказывает за любое лишнее срабатывание: 15% объектов чистые, и один
#   ложный дефект убирает объект из этого класса. Отсюда высокие пороги
#   у редких классов (`intersection` 0.70, `scale` 0.50).
# * **Геометрия даёт 10.5, изображения — 13.7** (слияние s+b; одиночный s — 13.5). Прирост сосредоточен в классах,
#   которые «видно»: `abstract`, `set`, `lowpoly`, `partial`. Геометрия
#   остаётся сильной на `noisy` (шум — свойство вершин).
# * **Остаются тяжёлыми** `intersection` (F1 0.06), `open` (0.29), `scale` (0.30),
#   `artifacts` (0.33) — редкие локальные дефекты. Карты внимания показывают, что
#   улика — 1–2 ячейки сетки, а ячейка `8×8` покрывает `64×64 px`: сигнал
#   размывается усреднением.
# * **Что делать дальше:** сетка `16×16`/`32×32` с attention-pooling вместо
#   усреднения (цель — `artifacts`/`open`), ансамбль `s+b` с калибровкой,
#   и только потом — увеличение самой модели: обучаемых параметров уже 7.7M при
#   8964 объектах.
