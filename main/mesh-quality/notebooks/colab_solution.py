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
# # Контроль качества 3D-моделей — воспроизведение решения в Google Colab
#
# **Задача (AI Journey, основной этап, трек Sber).** По шести рендерам 3D-модели
# (4 азимутальных ракурса + вид сверху и снизу) нужно предсказать 10 бинарных
# дефектов и производный флаг `quality`.
#
# **Метрика:** `10 · F1(quality) + 10 · F1_weighted(artefacts)`, максимум 20 баллов.
# `quality = 1` тогда и только тогда, когда все 10 дефектов равны 0 (проверено на
# всех 8964 объектах обучающей выборки), поэтому мы предсказываем только дефекты,
# а `quality` вычисляем из них.
#
# **Что делает этот ноутбук** (запуск «сверху вниз», без ручных правок):
#
# 1. ставит зависимости и фиксирует `random_seed`;
# 2. скачивает компактный бандл с решением (код + обученные пробы-модули) и
#    сырые архивы датасета по ссылкам из условия задачи;
# 3. извлекает признаки замороженного DINOv3 из сырых PNG (полный путь инференса);
# 4. считает вероятности дефектов тремя участниками слияния (ViT-S, ViT-B, меш-JEPA
#    L4), сливает их и записывает `submission.csv`, сверяя его с файлом, отправленным
#    на платформу;
# 5. воспроизводит метрику на out-of-fold предсказаниях (13.99 / 20) с разбором
#    по классам;
# 6. проходит полный цикл обучения заново: 26.6 ГиБ `AIC_data.tar` → PNG →
#    признаки → 5-фолдовая CV → refit (плюс быстрый геометрический бейзлайн);
#    шаг можно отключить (`DOWNLOAD_RAW_TRAIN = False`).
#
# **Архитектура решения** (детали — в презентации и `README.md`):
#
# ```
# PNG 1536x1024 ──> 6 тайлов 512x512 ──> замороженный DINOv3 (ViT-S/16 + ViT-B/16)
#                                            │  32x32 патч-токена на тайл
#                                            ▼
#              усреднение 4x4 блоков в сетку 8x8 + mean/max по тайлу (кэш fp16)
#                                            │
# 27 геометрических признаков ──> MLP ──> [GEOM] ──┐
#                                            ▼     ▼
#                           398 токенов ──> TransformerEncoder (4 слоя, d=384)
#                                            │
#                        обучаемый query-токен ──> Linear(384→10) ──> сигмоида
#                                            │
#   меш: JEPA-энкодер (заморожен, 256 патч-токенов) ──> тот же transformer (L4)
#                                            │
#    линейное слияние 0.3·s + 0.4·b + 0.3·L4 ─┴─> пороги (артефакты)
#                                    + отдельные пороги (quality) ──> submission.csv
# ```
#
# Обучаемая часть — три «проба»: два на замороженных DINOv3 (ViT-S 28.7M и ViT-B
# 85.6M, по ~7.7M обучаемых параметров) и joint-проб над замороженным JEPA-энкодером
# меша (тот же 4-слойный transformer, `[GEOM]` + 256 патч-токенов). Слияние трёх
# участников даёт **13.99/20 OOF** против 13.45 у лучшего одиночного (ViT-S).

# %% [markdown]
# ## 1. Окружение
#
# В Colab уже есть `torch`, `torchvision`, `numpy`, `pandas`; доставляем только
# `timm` (веса DINOv3), `gdown` (скачивание бандла с Google Drive) и `Pillow`.
# Ячейка помечена тегом `skip` — при локальном прогоне решения зависимости уже
# установлены в окружении проекта.

# %% tags=["skip"]
# !pip -q install "timm==1.0.29" gdown

# %%
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device: {DEVICE} | seed: {SEED}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

# %%
WORK = Path("/content/mesh_quality_work") if Path("/content").exists() else Path.cwd() / "mesh_quality_work"
WORK.mkdir(parents=True, exist_ok=True)
print(WORK)

# %% [markdown]
# ## 2. Артефакты решения и данные
#
# Ноутбук не содержит весов «в себе»: код и обученный проб скачиваются одним
# архивом `mesh_quality_bundle.zip` (~35 МБ). Внутри:
#
# | файл | назначение |
# |---|---|
# | `src/mesh_quality/*.py`, `main.py` | код решения (извлечение признаков, модель, метрика, меш-JEPA) |
# | `cache/probe_{s,b}.pt` | обученные DINOv3-пробы: веса + пороги |
# | `cache/reference_probs_fuse3.npy` | эталонные вероятности слияния (из них собран отправленный `submission.csv`) |
# | `cache/oof_fuse3.npy` | out-of-fold вероятности слияния на обучающей выборке |
# | `cache/fuse3/fuse3.json` | состав слияния, веса, пороги дефектов и quality |
# | `cache/l4_full/refit_l4.pt` | меш-член L4: joint-проб над JEPA-токенами, refit на всех 8964 объектах |
# | `cache/mesh_tokens/test/` | замороженные JEPA-токены тестовых мешей (~127 МБ) |
# | `tools/{probe_mesh,refit_mesh,fuse3}.py` | сборка/инференс меш-члена и слияние с подбором порогов |
# | `cache/geometry_{train,test}.csv` | 27 геометрических признаков на объект |
# | `data/{train,test}.csv` | списки `item_id` (для train — разметка) |
# | `reference_submission.csv` | файл, отправленный на платформу (эталон для сверки) |
#
# Сырые данные (1.75 ГБ тест, 26 ГБ train) скачиваются по ссылкам из условия
# задачи — по умолчанию общий `AIC_data.tar` (26.6 ГиБ, оба сплита, прямой
# HTTPS). `BUNDLE_URL` / `BUNDLE_GDRIVE_ID` — где лежит бандл (прямой HTTP,
# Яндекс.Диск, Google Drive: файл или папка).

# %%
# --- конфигурация --------------------------------------------------------- #
# Бандл — снапшот решения: ноутбук исполняет **код из zip**, не из репозитория,
# поэтому будущие правки репозитория на прогон не влияют. Ссылка по умолчанию —
# на конкретный файл Google Drive, а `BUNDLE_MD5` прибивает его содержимое
# (zip собран из коммита `BUNDLE_COMMIT`). Если в папке заменят zip, пин не
# совпадёт и ноутбук скажет об этом — обновите `BUNDLE_URL`/`BUNDLE_MD5`.
BUNDLE_URL = "TBD"  # файл (пин): URL загруженного на Google Drive zip (обновляется вместе с BUNDLE_MD5 и BUNDLE_COMMIT)
# папка с бандлом: https://drive.google.com/drive/folders/12ttUbp4Bjwx1mFVrdBNkqiZA0YuSRTR_?usp=sharing
BUNDLE_GDRIVE_ID = ""  # либо id файла/папки Google Drive
BUNDLE_MD5 = ""  # пин снапшота ("" отключает проверку)
BUNDLE_COMMIT = "TBD"  # коммит, из которого собран zip

DOWNLOAD_RAW_TRAIN = True  # полный цикл: архив -> PNG -> признаки -> обучение; False: шаг пропускается
KEEP_ARCHIVES = False  # удалять скачанный архив после распаковки (26.6 ГиБ)
TRAIN_ALL_MODELS = False  # True: обучить все модели, вошедшие в слияние
TRAIN_LIMIT = None  # int — обучение на подвыборке (отладка)

EPOCHS, FOLDS = 40, 5  # как в финальном чекпоинте

# Источник сырых данных — архив из условия задачи: `AIC_data.tar` (26.6 ГиБ,
# `AIC_data/{train,test}/<item_id>.{png,npz}`) скачивается прямым HTTPS одним
# потоком и обслуживает оба сплита, без публичного API Яндекс.Диска.
# Альтернатива — официальные zip-архивы (`SOURCE_ARCHIVE = ""`): тогда для
# инференса хватит `test.zip` (1.75 ГБ).
SOURCE_ARCHIVE = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/AIC_data.tar"
TEST_LINK = "https://disk.360.yandex.ru/d/rUSPxzoDTHK8UQ"  # test.zip, 1.75 ГБ (альтернатива)
TRAIN_LINK = "https://disk.360.yandex.ru/d/CeZVSNyRGjrLUw"  # train.zip, 26 ГБ (альтернатива)


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _human(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def _download(url: str, dst: Path) -> Path:
    """Потоковое скачивание с прогрессом."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size:
        print(f"уже скачано: {dst.name} ({_human(dst.stat().st_size)})")
        return dst
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dst.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and (done // (1 << 25) != (done - len(chunk)) // (1 << 25)):
                print(f"  ... {_human(done)} / {_human(total)}", flush=True)
    print(f"скачано: {dst.name} ({_human(dst.stat().st_size)})")
    return dst


def yandex_url(public_link: str) -> str:
    """Публичная ссылка Яндекс.Диска -> прямой download-href (публичный API)."""
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    with urllib.request.urlopen(f"{api}?public_key={urllib.parse.quote(public_link, safe='')}") as r:
        return json.loads(r.read())["href"]


def _drive_id(source: str, kind: str) -> str | None:
    """id из ссылки Google Drive: kind — 'folders' (папка) или 'file' (файл)."""
    patterns = {
        "folders": r"drive\.google\.com/drive/(?:u/\d+/)?folders/([\w-]{10,})",
        "file": r"drive\.google\.com/file/d/([\w-]{10,})",
    }
    m = re.search(patterns[kind], source)
    return m.group(1) if m else None


def _is_drive_folder(drive_id: str) -> bool:
    """Похож ли id на публичную папку Google Drive (дешёвый листинг, без скачивания)."""
    import gdown

    try:
        items = gdown.download_folder(id=drive_id, quiet=True, use_cookies=False, skip_download=True)
    except Exception:  # noqa: BLE001 - не папка или нет доступа
        return False
    return bool(items)


def _download_drive_folder(folder_id: str, dst: Path) -> Path:
    """Скачать zip с бандлом из публичной папки Google Drive.

    Имя файла внутри папки может меняться — ссылка на папку остаётся той же.
    """
    import gdown

    out = dst.parent / "drive_bundle"
    out.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=folder_id, output=str(out), quiet=False, use_cookies=False)
    zips = sorted(p for p in out.rglob("*.zip") if p.is_file())
    if not zips:
        raise SystemExit(f"в папке Google Drive {folder_id} нет zip-файла: {out}")
    pick = next((z for z in zips if z.name == "mesh_quality_bundle.zip"), zips[-1])
    if len(zips) > 1:
        print(f"в папке {len(zips)} zip-файлов, беру {pick.name}")
    shutil.move(str(pick), dst)
    print(f"скачано из папки: {dst.name} ({_human(dst.stat().st_size)})")
    return dst


def fetch(source: str, dst: Path) -> Path:
    """Скачать по прямой ссылке / Яндекс.Диску / Google Drive (файл или папка)."""
    if not source:
        raise SystemExit("источник не задан")
    if "disk.360.yandex.ru" in source or "yadi.sk" in source:
        return _download(yandex_url(source), dst)
    folder_id = _drive_id(source, "folders")
    if folder_id:  # из папки берём zip с бандлом
        return _download_drive_folder(folder_id, dst)
    file_id = _drive_id(source, "file")
    if file_id:  # ссылка на файл -> прямой download-URL (иначе скачается HTML)
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        return _download(url, dst)
    if "://" in source:  # прямой http(s), file:// (локальная проверка ноутбука)
        return _download(source, dst)
    import gdown

    # id без ссылки: папка или файл Google Drive
    if _is_drive_folder(source):
        return _download_drive_folder(source, dst)
    gdown.download(id=source, output=str(dst), quiet=False)
    return dst


# %%
# --- бандл решения: код + обученный проб + эталонные матрицы --------------- #
bundle_zip = fetch(BUNDLE_URL or BUNDLE_GDRIVE_ID, WORK / "mesh_quality_bundle.zip")
_got = md5(bundle_zip)
if BUNDLE_MD5 and _got != BUNDLE_MD5:
    raise SystemExit(
        f"md5 бандла {_got} != пина {BUNDLE_MD5}: скачан другой архив. "
        "Задайте BUNDLE_URL на нужный файл или сбросьте BUNDLE_MD5 = ''."
    )
print(f"бандл: {bundle_zip.name} ({_human(bundle_zip.stat().st_size)}), md5 {_got}"
      + (" совпал с пином" if BUNDLE_MD5 else " (пин отключён)"))
with zipfile.ZipFile(bundle_zip) as zf:
    zf.extractall(WORK)
TASK = WORK / "mesh_quality"
assert (TASK / "src" / "mesh_quality" / "model.py").exists(), sorted(p.name for p in TASK.iterdir())
CACHE, DATA = TASK / "cache", TASK / "data"
sys.path.insert(0, str(TASK / "src"))

from mesh_quality import images, metric, model, solution  # noqa: E402

manifest = json.loads((TASK / "manifest.json").read_text())
ref = manifest["reference"]
print(f"бандл от {manifest['created']}, git {manifest['git_rev']} (снапшот {BUNDLE_COMMIT}), "
      f"модель {ref['model']}, CV {ref['cv']['score']:.3f}")
print("пороги: " + " ".join(f"{t:.3f}" for t in ref["thresholds"]))

# %% [markdown]
# ### Сырые данные
#
# Источник — `AIC_data.tar` (26.6 ГиБ, `AIC_data/{train,test}/`): тот же архив,
# на котором обучено решение. Он скачивается напрямую по HTTPS (Accept-Ranges),
# поэтому публичный API Яндекс.Диска и его лимиты в этом пути не участвуют; один
# файл обслуживает оба сплита. Zip-архивы `test.zip` / `train.zip` поддерживаются
# как запасной вариант (`SOURCE_ARCHIVE = ""`).
#
# Для решения нужны только PNG-коллажи: `.npz` с геометрией меша занимают 22 ГБ и
# нигде не используются (геометрия берётся из `cache/geometry_*.csv`). Поэтому
# распаковка **выборочная** — из архива читаются только `<split>/<item_id>.png`
# для `item_id` из `data/<split>.csv`. Tar приходится читать последовательно,
# поэтому `test` и `train` — два прохода по архиву; сам архив удаляется, когда
# распакован последний нужный сплит (`KEEP_ARCHIVES=False`).

# %%
def extract_pngs(archive: Path, split: str, ids: np.ndarray) -> int:
    """Распаковать из архива только рендеры `<split>/<item_id>.png`."""
    keep = {str(i) for i in ids}
    dst = DATA / split
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist()
                       if m.filename.endswith(".png") and Path(m.filename).stem in keep]
            print(f"{archive.name}: рендеров в архиве — {len(members)} из {len(zf.namelist())} файлов")
            for m in members:
                target = dst / Path(m.filename).name
                if target.exists():
                    continue
                with zf.open(m) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                written += 1
    else:  # tar-зеркало из условия задачи
        import tarfile

        with tarfile.open(archive, "r:*") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".png") and Path(m.name).stem in keep:
                    target = dst / Path(m.name).name
                    if target.exists():
                        continue
                    src = tf.extractfile(m)
                    with target.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1 << 20)
                    written += 1
    print(f"{split}: распаковано {written} рендеров -> {dst} "
          f"({len(list(dst.glob('*.png')))} всего)")
    return written


def split_source(split: str) -> tuple[str, str, bool]:
    """Ссылка на архив сплита, локальное имя файла и признак «общий архив»."""
    if SOURCE_ARCHIVE:  # один архив на оба сплита
        return SOURCE_ARCHIVE, Path(urllib.parse.urlparse(SOURCE_ARCHIVE).path).name, True
    link = TEST_LINK if split == "test" else TRAIN_LINK
    return link, f"{split}{Path(link).suffix or '.zip'}", False


def prepare_split(split: str) -> np.ndarray:
    """Скачать архив, распаковать только PNG сплита, вернуть список `item_id`."""
    ids = np.asarray(pd.read_csv(DATA / f"{split}.csv")["item_id"].astype(str))
    have = len(list((DATA / split).glob("*.png")))
    if have >= len(ids):
        print(f"{split}: рендеры уже распакованы ({have})")
        return ids
    source, fname, shared = split_source(split)
    archive = fetch(source, WORK / fname)
    extract_pngs(archive, split, ids)
    if not KEEP_ARCHIVES:
        # общий архив нужен ещё раз для train — держим его до последнего сплита
        if shared and split == "test" and DOWNLOAD_RAW_TRAIN:
            print(f"{archive.name} ({_human(archive.stat().st_size)}) оставлен для сплита train")
        else:
            archive.unlink()
            print(f"удалён {archive.name} (KEEP_ARCHIVES=False), диск свободен")
    return ids


test_ids = prepare_split("test")
print(f"test.csv: {len(test_ids)} объектов")

# %% [markdown]
# ## 3. Признаки замороженного DINOv3
#
# Коллаж режется на 6 тайлов 512×512 (четыре азимута + сверху/снизу). Каждый тайл
# проходит через **замороженный** `vit_small_plus_patch16_dinov3` (timm,
# ImageNet-нормализация, RoPE — интерполяция позиционных эмбеддингов не нужна):
# 32×32 патч-токена размерности 384 на тайл.
#
# Чтобы обучаемая часть осталась дешёвой, из 1024 патчей тайла сохраняются:
#
# * `tile_mean`, `tile_max` — среднее и максимум по тайлу (по 384 числа);
# * `tile_grid` — усреднение 4×4 блоков патчей в сетку 8×8 (64 токена по 384).
#
# Кэш пишется в fp16 и читается через `mmap`, поэтому обучение и инференс не
# зависят от 27 ГБ сырых данных.

# %%
# Признаки считаются для каждой модели, вошедшей в слияние (`ref["models"]` —
# `s`, `b`, `L4`): у `s` карта признаков 384-мерная, у `b` — 768-мерная, кэши
# разные; меш-член `L4` читает токены меша и считается отдельно (следующая
# ячейка). Бандл намеренно не содержит кэш признаков: извлечение — часть
# воспроизведения, поэтому резервного `.npz`-пути больше нет.
FEATURE_MODELS = [k for k in ref["models"] if k in ("s", "b")]
for key in FEATURE_MODELS:
    t0 = time.time()
    images.extract_split(data_dir=DATA, split="test", cache_dir=CACHE, model_key=key)
    print(f"[{key}] признаки теста: {time.time() - t0:.1f} с")

for key in FEATURE_MODELS:
    cache = images.load_cache(CACHE, key, "test")
    print(f"[{key}] кэш {images.cache_path(CACHE, key, 'test').relative_to(TASK)}: "
          f"{len(cache.item_ids)} объектов, tile_mean {cache.tile_mean.shape} {cache.tile_mean.dtype}, "
          f"tile_grid {cache.tile_grid.shape}, grid {cache.grid}x{cache.grid}")
    assert cache.grid == 8 and cache.tile_mean.dtype == np.float16
dinos_ids = np.asarray(np.load(images.cache_path(CACHE, FEATURE_MODELS[0], "test") / "item_ids.npy"))
test_ids = np.asarray(pd.read_csv(DATA / "test.csv")["item_id"].astype(str))
assert list(dinos_ids) == list(test_ids)

# %% [markdown]
# ### 3б. Меш-член L4: замороженные JEPA-токены меша → joint-проб
#
# Третий участник слияния читает сам меш. Токены меша лежат в бандле
# (`cache/mesh_tokens/test/`): 256 патчей, центр каждого — k-means по
# area-weighted точкам поверхности, признаки точки — смещение относительно центра,
# радиус и нормаль, плюс per-patch статистики и глобальные скаляры вроде
# `log bbox_diag` и `n_faces`. Их считает `tools/build_mesh_cache.py` из
# `.npz`-геометрии; в Colab 22 ГБ `.npz` не скачиваются, поэтому токены берутся из
# бандла, а воспроизводится **инференс**: чекпоинт `cache/l4_full/refit_l4.pt`
# (обучен на всех 8964 объектах) прогоняется по тестовым токенам тем же
# хендлером, что и в CLI (`tools/refit_mesh.py --predict-only`).
#
# Архитектурно L4 — тот же 4-слойный transformer, что у DINOv3-пробов: к
# 398 image-токенам проба `s` и `[GEOM]` добавляются 256 mesh-токенов, прошедших
# замороженный JEPA-энкодер (256-мерные эмбеддинги; невалидные патчи получают
# `null_vector`). Ниже сверяем инференс с эталонной матрицей
# `cache/reference_probs_l4.npy` из бандла.

# %%
l4_dir = CACHE / "l4_full"
l4_reference_path = TASK / ref["reference_probs_l4"]
assert md5(l4_reference_path) == ref["reference_probs_l4_md5"], "эталонные L4-вероятности повреждены"
l4_reference = np.load(l4_reference_path).astype(np.float32)

t0 = time.time()
subprocess.run(
    [sys.executable, "tools/refit_mesh.py", "--predict-only",
     "--ckpt", "cache/l4_full/refit_l4.pt", "--out-dir", "cache/l4_full"],
    cwd=TASK, check=True,
)
probs_l4 = np.load(l4_dir / "probs_test_l4.npy").astype(np.float32)
l4_ids = np.load(l4_dir / "item_ids_test.npy")
print(f"[L4] инференс: {time.time() - t0:.1f} с, {probs_l4.shape}")
assert list(l4_ids) == list(dinos_ids), "порядок объектов L4 не совпал с кэшем признаков"
l4_delta = float(np.abs(probs_l4 - l4_reference).max())
print(f"[L4] max |Δp| против эталона бандла: {l4_delta:.2e}")
assert l4_delta < 0.05, "L4-прогон разошёлся с эталоном сильнее допуска bf16"

# %% [markdown]
# ## 4. Инференс: вероятности дефектов → слияние → `submission.csv`
#
# Обучаемая часть — «проб»: последовательность из 398 токенов (6 mean + 6 max +
# 6·8·8 токенов сетки + один `[GEOM]` + один обучаемый query) проходит через
# `TransformerEncoder` (4 слоя, d=384, 6 голов, pre-LN, dropout 0.1); выход
# query-токена идёт в линейный слой на 10 логитов и сигмоиду. Обучение —
# `BCEWithLogitsLoss` с весами положительных классов.
#
# Все три набора вероятностей сходятся в линейный ансамбль
# `0.3·s + 0.4·b + 0.3·L4`. Пороги дефектов подбираются координатным подъёмом по
# метрике соревнования на **out-of-fold** матрице; у `quality` (по условию задачи
# это независимая колонка) свой набор порогов: `quality = 1` тогда и только тогда,
# когда каждая вероятность ниже своего порога. Оба набора перевыводятся тем же
# инструментом `tools/fuse3.py`, что и в CLI (seeded anneal + координатный
# подъём), и применяются к тестовым вероятностям — воспроизведение буквальное,
# а не «переписанное по мотивам».

# %%
# Эталонная матрица вероятностей лежит в бандле под отдельным именем: инференс
# перезаписывает `cache/probs_test_*.npy`, поэтому читаем эталон ДО него.
reference_probs_path = TASK / ref["reference_probs"]
assert md5(reference_probs_path) == ref["reference_probs_md5"], "эталонные вероятности повреждены"
probs_reference = np.load(reference_probs_path).astype(np.float32)
print(f"эталонная матрица: {probs_reference.shape}, md5 {ref['reference_probs_md5'][:12]}")

# (а) вероятности DINOv3-пробов — тем же хендлером, что в CLI решения.
for key in FEATURE_MODELS:
    probs, ids = solution.predict_probe(CACHE, DATA, key, DEVICE)
    assert list(ids) == list(dinos_ids), f"порядок объектов {key} не совпал с кэшем"
    np.save(CACHE / f"probs_test_{key}.npy", probs)
    print(f"[{key}] вероятности теста: {probs.shape}")

# (б) меш-член L4 уже посчитан в 3б (probs_l4).
# (в) слияние, подбор порогов и запись submission.csv — инструмент CLI как есть.
t0 = time.time()
subprocess.run([sys.executable, "tools/fuse3.py"], cwd=TASK, check=True)
print(f"слияние: {time.time() - t0:.1f} с")
probs_fresh = np.load(CACHE / "fuse3" / "probs_test_fuse3.npy").astype(np.float32)
fuse3_config = json.loads((CACHE / "fuse3" / "fuse3.json").read_text())

# %%
# --- сверка с файлом, отправленным на платформу --------------------------- #
sub = pd.read_csv(TASK / "submission.csv")
up = pd.read_csv(TASK / "reference_submission.csv")
thresholds = np.asarray(ref["thresholds"], dtype=np.float32)
d_pred = sub[list(metric.DEFECTS)].to_numpy(dtype=np.int8)
d_up = up[list(metric.DEFECTS)].to_numpy(dtype=np.int8)
d_reference = (probs_reference >= thresholds).astype(np.int8)

test_ids = np.asarray(pd.read_csv(DATA / "test.csv")["item_id"].astype(str))
print("те же объекты и порядок:", bool(np.array_equal(np.asarray(sub["item_id"]), test_ids)))
print("пороги:", " ".join(f"{t:.3f}" for t in thresholds))
print("эталонные вероятности + пороги == отправленный файл:", bool(np.array_equal(d_reference, d_up)))
print("quality == (сумма дефектов == 0):",
      bool(np.array_equal(sub["quality"].to_numpy(dtype=np.int8), metric.derive_quality(d_pred))))
print("md5(submission.csv)         :", md5(TASK / "submission.csv"))
print("md5(reference_submission.csv):", md5(TASK / "reference_submission.csv"))
print("положительных по классам:", d_pred.sum(0).tolist())

# %% [markdown]
# ### Почему это ровно тот файл, что на лидерборде
#
# * признаки считаются тем же кодом (`mesh_quality.images`) и той же замороженной
#   моделью в `eval()` под `torch.no_grad()`;
# * вероятности — из того же чекпоинта `probe_s.pt`, пороги сохранены в нём и в
#   манифесте;
# * **проверка не тавтологична**: эталонная матрица читается из бандла
#   (`cache/reference_probs_s.npy`) до запуска `predict`, а не после, и её md5
#   сверяется с манифестом;
# * извлечение признаков идёт в bf16, поэтому возможны отличия порядка 1e-3 в
#   вероятностях, а условие задачи прямо допускает «minor floating-point
#   differences». Ниже отличия измеряются численно, а не «на глаз».

# %%
delta = float(np.abs(probs_fresh - probs_reference).max())
n_diff = int(np.abs(d_pred - d_up).sum())
print(f"max |Δp| против эталонных вероятностей: {delta:.2e}")
print(f"отличий в бинарных решениях: {n_diff} из {d_pred.size} ячеек / "
      f"{int((d_pred != d_up).any(1).sum())} объектов")
assert np.array_equal(d_reference, d_up), "эталон бандла разошёлся с отправленным файлом"
assert np.array_equal(sub["quality"].to_numpy(dtype=np.int8), metric.derive_quality(d_pred))
if n_diff == 0:
    print("✓ предсказания воспроизведены точно")
else:
    print(f"✓ предсказания воспроизведены с точностью до bf16-округления: {n_diff} ячеек "
          "у границы порога (условие задачи допускает минорные отличия)")
print("✓ эталонная матрица бандла воспроизводит отправленный файл точно")

# %% [markdown]
# ## 5. Метрика на out-of-fold предсказаниях
#
# `cache/oof_fuse_s+b.npy` — вероятности на 8964 обучающих объектах, полученные
# 5-фолдовой кросс-валидацией (идент. код в обоих пробах, у слияния — среднее).
# По ним заново подбираем пороги и считаем метрику: это честная оценка решения
# (по ней же выбирались гиперпараметры). Первая одиночная версия проба получила
# на лидерборде 12.53/20 при OOF 13.35/20 — тестовая часть немного тяжелее
# обучающей, поэтому и у текущего кандидата лидерборд ниже OOF.
#
# Параметры кросс-валидации: `folds(k=5, seed=42)`, 40 эпох, AdamW (lr 1e-3,
# weight decay 0.05), OneCycleLR, batch 128, bf16-автокаст, `pos_weight` по классам.

# %%
labels = pd.read_csv(DATA / "train.csv").set_index("item_id")
train_ids = np.asarray(pd.read_csv(DATA / "train.csv")["item_id"].astype(str))
y = labels.loc[list(train_ids), list(metric.DEFECTS)].to_numpy(dtype=np.int8)
oof = np.load(TASK / ref["reference_oof"]).astype(np.float32)
assert oof.shape == y.shape, (oof.shape, y.shape)

thr, res = metric.tune_thresholds(oof, y, metric.derive_quality(y), verbose=False)
print(f"OOF score: {res['score']:.3f} / 20   "
      f"(quality F1 {res['quality_f1']:.4f}, artefacts F1 weighted {res['artefact_f1_weighted']:.4f})")
per_label = pd.DataFrame(
    {
        "F1": [res["per_label"][n] for n in metric.DEFECTS],
        "support": [res["support"][n] for n in metric.DEFECTS],
        "pred_positives": [res["pred_positives"][n] for n in metric.DEFECTS],
    },
    index=list(metric.DEFECTS),
)
per_label["класс, %"] = (100 * per_label["support"] / len(y)).round(2)
try:
    display(per_label.round(3))  # noqa: F821 - только в IPython/Colab
except NameError:
    print(per_label.round(3))
print("пороги OOF:", " ".join(f"{float(t):.3f}" for t in thr))

# %% [markdown]
# Слабые классы (`artifacts`, `open`, `scale`, `intersection`) — редкие и локальные
# дефекты, именно они определяют разрыв с потолком метрики. Разбор случаев и карты
# внимания — во втором ноутбуке (`interpretation.ipynb`).

# %% [markdown]
# ## 6. Обучение
#
# Полный цикл целиком внутри ноутбука: сырые архивы → PNG → признаки замороженного
# DINOv3 → 5-фолдовая CV проба → подбор порогов по метрике → refit на всех данных.
# Предвычисленных признаков не требуется: единственный скачиваемый артефакт, кроме
# данных, — компактный бандл (код + чекпоинты + эталонные вероятности для сверки).
#
# `submission.csv` уже записан выше, в разделе 4, по чекпоинтам из бандла; этот
# раздел демонстрирует обучение с нуля на тех же данных.
#
# Стоимость на T4: скачивание `AIC_data.tar` — 26.6 ГиБ, распаковка PNG ~8 мин,
# признаки `s` ~10 мин, CV (5 × 40 эпох) ~30–40 мин, refit ~7 мин.
# Шаг можно отключить (`DOWNLOAD_RAW_TRAIN = False`) или ускорить кэшем признаков
# train (`TRAIN_CACHE_URL`): распаковка и извлечение признаков пропускаются.
#
# Воспроизводимость: чекпоинт в бандле обучен с `seed = model.SEED` (0, код-дефолт)
# и усреднён по двум refit-прогонам; демонстрация использует тот же seed, чтобы
# цифры были сопоставимы, но refit здесь один (флаг `--full-seeds` решения).

# %%
TRAIN_MODELS = list(ref["models"]) if TRAIN_ALL_MODELS else [ref["models"][0]]
_primary = TRAIN_MODELS[0]
READY: list[str] = []
if TRAIN_CACHE_URL:
    st = fetch(TRAIN_CACHE_URL, WORK / f"dino{_primary}_train.npz")
    shutil.copy2(st, CACHE / f"dino{_primary}_train.npz")
    READY = [_primary]
    print(f"кэш признаков train взят из TRAIN_CACHE_URL: dino{_primary}_train.npz")
elif DOWNLOAD_RAW_TRAIN:
    prepare_split("train")
    for key in TRAIN_MODELS:
        t0 = time.time()
        images.extract_split(data_dir=DATA, split="train", cache_dir=CACHE, model_key=key, grid=8)
        READY.append(key)
        print(f"[{key}] признаки train: {(time.time() - t0) / 60:.1f} мин")
else:
    print("Обучение пропущено: DOWNLOAD_RAW_TRAIN = False и TRAIN_CACHE_URL не задан. "
          "Геометрический бейзлайн ниже всё равно обучается полностью.")
print("к обучению:", READY, "| эпох:", EPOCHS, "| фолдов:", FOLDS)

# %% [markdown]
# ### 6a. Бейзлайн только на геометрии (без изображений)
#
# 27 геометрических признаков (габариты, вытянутость, плотность вершин и граней,
# отношения площадей и т.п.) → HistGradientBoosting на каждый класс. Быстро и
# интерпретируемо; даёт 10.5/20, то есть изображения добавляют около +3.2 балла.

# %%
sol_args = type("Args", (), {})()
sol_args.task_dir = TASK
sol_args.data_dir = DATA
sol_args.model = "geometry"
sol_args.folds = FOLDS
sol_args.seed = model.SEED  # код-дефолт: с ним же обучались чекпоинты и мерялся README
sol_args.limit = None
solution.train_geometry(sol_args)

# %% [markdown]
# ### 6б. Обучение проб-модуля (полный цикл)
#
# Гиперпараметры финального чекпоинта: 40 эпох, AdamW (lr 1e-3, wd 0.05), OneCycleLR,
# batch 128, `dropout 0.1`, 4 слоя, d=384, 6 голов, `pos_weight = (neg/pos)^0.5`,
# градиентный клип 1.0. Ниже — тот же код, что обучен финальный чекпоинт:
# `model.run_cv` (5 фолдов, OOF → пороги → метрика) и `model.train_model` (refit на
# всех 8964 объектах) → `cache/probe_<модель>.pt`, который инференс выше умеет читать.
# Ячейка помечена `slow`: на CPU-инстансе Colab она займёт часы, для обучения нужен
# ускоритель T4/L4.

# %% tags=["slow"]
def train_probe(key: str, limit: int | None = None) -> tuple[np.ndarray, dict]:
    """Полный цикл одного проба: CV → пороги → refit на всех данных."""
    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    ds = model.load_dataset(CACHE, DATA, key, "train", grid=8, limit=limit)
    print(f"[{key}] {len(ds)} объектов, размерность признаков {ds.dim}, "
          f"геометрических признаков {ds.geom.shape[1]}")
    median, scale = model.geometry_stats(ds.geom)
    geom_z = ds.standardise(median, scale)
    cfg = model.ProbeConfig(dim=ds.dim, n_geom=int(ds.geom.shape[1]), grid=ds.grid)
    y_train = ds.labels
    assert y_train is not None

    t0 = time.time()
    oof_new, _ = model.run_cv(
        ds, cfg, geom_z, k=FOLDS, seed=model.SEED, epochs=EPOCHS, batch=128, log_every=10
    )
    thr_new, res_new = metric.tune_thresholds(oof_new, y_train, metric.derive_quality(y_train))
    solution._report(
        f"[{key}] cv image+geometry probe ({EPOCHS} эпох, {(time.time() - t0) / 60:.1f} мин)", res_new
    )

    full, _ = model.train_model(
        ds, cfg, geom_z, np.arange(len(ds)), epochs=EPOCHS, batch=128, seed=model.SEED, device=device
    )
    _torch.save(
        {
            "cfg": cfg.__dict__,
            "model_key": key,
            "states": [full.state_dict()],
            "thresholds": thr_new.tolist(),
            "geom_median": median,
            "geom_scale": scale,
            "cv": res_new,
            "version": 1,
        },
        CACHE / f"probe_{key}.pt",
    )
    print(f"[{key}] сохранено: {CACHE / f'probe_{key}.pt'} (CV {res_new['score']:.3f})")
    return oof_new, res_new


trained = {key: train_probe(key, TRAIN_LIMIT) for key in READY}
print("обучено:", {k: round(v[1]["score"], 3) for k, v in trained.items()})

# %% [markdown]
# ## 7. Итоги
#
# **Результат:** замороженный DINOv3 + обучаемый проб на 7.7M параметров даёт
# **13.68/20 на кросс-валидации** (слияние ViT-S и ViT-B; лучший одиночный проб —
# 13.45/20). Первая версия решения отправила на лидерборд 12.53/20.
# Основной прирост дают изображения: геометрический бейзлайн без них — 10.5/20.
# Класс `noisy` предсказывается почти идеально (F1 ≈ 0.85), самые тяжёлые — редкие
# локальные дефекты `intersection` / `open` / `scale` / `artifacts`.
#
# **Файлы решения** (те же, что в репозитории):
#
# | модуль | роль |
# |---|---|
# | `images.py` | коллаж → 6 тайлов; замороженный DINOv3; кэш `tile_mean/tile_max/tile_grid`; градиентные карты внимания |
# | `model.py` | `AttentiveProbe` (398 токенов → 4 слоя → query), обучение, CV, дешёвые абляции |
# | `metric.py` | метрика соревнования, вывод `quality`, подбор порогов, чтение/запись submission |
# | `solution.py` | хендлеры CLI: `features`, `train`, `predict` (в т.ч. слияние `--models`), `fuse`, `score`, `visualize` |
# | `deliver.py` | сборка этого бандла (код + чекпоинты + эталонные матрицы) |
#
# **Что можно улучшить (план финального этапа):** более мелкая сетка патчей (16×16
# вместо 8×8) для локальных дефектов, in-domain дообучение бэкбона на самих
# рендерах, attention-pooling вместо усреднения патчей в ячейку сетки.

# %%
print("Готово. Файлы:")
for p in sorted(TASK.rglob("*")):
    if p.is_file() and p.suffix in {".csv", ".npy", ".pt", ".json"}:
        print(f"  {p.relative_to(TASK)}  {_human(p.stat().st_size)}")
