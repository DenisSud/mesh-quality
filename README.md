# 3D Mesh Quality Control

Detect defects on 3D meshes and classify every object as good or bad. Solution
for the **AI Challenge** main-stage task ([aiijc.com](https://aiijc.com/), Sber
AI generative-models area).

**Metric:** `10·F1(quality) + 10·F1_weighted(defects)` (out of 20).
**Result:** fused probe **OOF 13.68 · public LB 13.86** — best single backbone
13.45, geometry-only baseline 10.52.

| variant | OOF /20 | artefact F1w | quality F1 |
|---|---|---|---|
| geometry-only HistGB | 10.52 | 0.557 | 0.495 |
| DINOv3-s probe | 13.45 | 0.671 | 0.675 |
| DINOv3-b probe | 13.30 | 0.664 | 0.666 |
| **fused s+b (shipped)** | **13.68** | **0.685** | **0.684** |

## Approach

Frozen DINOv3 patch features from 6 rendered views (8×8 pooled grid + per-tile
mean/max summaries) + 27 geometry features → a 4-layer attentive probe
(7.7M trainable, 398 tokens, d=384) → 10 sigmoid defect heads. `quality` is
derived as the all-zero rule, so it is thresholded *after* the defect heads.
Per-label thresholds are tuned on OOF probabilities by coordinate ascent
directly on the competition metric; the final submission fuses two backbones
(ViT-S + ViT-B) by averaging test probabilities.

A full 5-fold CV ablation matrix showed token resolution and pooling statistics
move nothing (all variants within ±0.1 of the shipped 8×8 + summaries recipe) —
head capacity is the only lever that mattered. Threshold tuning alone is worth
≈ +0.4 over the 0.5 default.

## Layout

```
main/mesh-quality/
  main.py            CLI: train | predict | score | visualize | features |
                     resubmit | fuse | logscore | deliver
  src/mesh_quality/  geometry + DINOv3 features, attentive probe, metric and
                     threshold tuning, solution handlers, Colab bundle staging
  tests/             CPU pytest suite for the pipeline seams
  notebooks/         jupytext sources for the two graded notebooks
  presentation/      Typst deck + figures
  tools/             ablations, figures, cache migration, data explorer
  submissions.csv    leaderboard log (per-upload probability matrices included)
docs/tasks/mesh-quality.md   full task spec
src/aiijc/                   shared task-CLI scaffold
```

## Quickstart

Everything runs inside the shared devenv:

```bash
devenv shell
uv run pytest -q                       # CPU suite, no GPU needed

# rebuild the feature cache from the PNGs, then the anchor CV (~13.47 ±0.1)
uv run python main/mesh-quality/main.py features --model s
uv run python main/mesh-quality/tools/hyperparams.py g8summ10

# geometry-only fallback (10.52 at seed 42)
uv run python main/mesh-quality/main.py train --model geometry --seed 42

# stage the Colab bundle
uv run python main/mesh-quality/main.py deliver --models s,b --ref-model s+b
```

## Data

The dataset is **not** committed (`data/` is gitignored; only the small label
CSVs are kept). Download links are in
[`docs/tasks/mesh-quality.md`](docs/tasks/mesh-quality.md): 8964 train / 669
test objects, each a 6-view PNG collage (1536×1024) plus a mesh `.npz`.

## Notes

- The submission artifacts are reproducible: `submissions.csv` logs every upload
  with commit, thresholds and md5, and `submissions/*_probs.npy` /
  `*_oof.npy` keep the probability matrices, so any uploaded CSV is regenerable
  via `resubmit`.
- Three pipeline bugs found and fixed along the way, all the same class —
  a check that can silently disagree with what it claims to verify: a stale
  checkpoint whose token assembly predated a code fix, an experiment log whose
  failed append wiped its own history, and a reproduction check that read the
  file it had just overwritten. Each fix is pinned by a test.
- Measured facts, ablations and post-mortems: task README and
  [`docs/tasks/mesh-quality.md`](docs/tasks/mesh-quality.md).
