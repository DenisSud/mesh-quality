# Task 1 — 3D Mesh Quality Control

*3D Mesh Quality Control: автоматизация поиска дефектов для обучения генеративных моделей.*

**Organizer / area:** Sber (SBER AI), generative models.
**ML problem:** multi-label defect classification + binary quality classification.
**Metric:** `f1_final = 10·f1_score(y_quality) + 10·f1_score(y_artefacts, average="weighted")`
**Code:** [`main/mesh-quality/`](../../main/mesh-quality/README.md)

Datasets of 3D objects for generative/compression/analysis models must contain
only good meshes — smooth surfaces, no defects — and must not be dominated by
overly simple objects, which hurt generalization of large pretrains. The task is
to detect artifacts automatically on meshes and/or their renders and to provide
a visualization that explains each decision (per-region analysis, attention maps
on renders, etc.).

For every object (`item_id`) predict:

- `quality` — 1 = good, 0 = bad (any defect makes the object bad);
- the list of defects present (multi-label).

## Defects

| Label | Meaning |
|---|---|
| `simple` | too simple: a geometric primitive (cube, sphere, cylinder, triangle), flat object, extruded logo/image |
| `lowpoly` | polygon count lower than needed for the detail; angles visible where a smooth surface is expected |
| `noisy` | surface consists of noisy points/polygons (typical for 3D scans) |
| `artifacts` | strange transitions between triangles, intersections, broken look |
| `abstract` | 3D text, graphs, tables, diagrams, charts, 3D-printing supports, Minecraft-style objects; no single coherent object |
| `scale` | the object takes too little of the frame or the scale is not optimal; details indistinguishable |
| `partial` | object/scene not visible from some angles (hidden by a wall, fence, plane); e.g. closed by a wall on one side, internal details visible only from one angle |
| `set` | several unrelated parts, or an object separated in space; logically related objects (racket + ball) are not a defect unless there are many |
| `open` | hollow inside or built from flat surfaces without volume; e.g. flat scan, a plane covering the object, two nearly empty angles |
| `intersection` | objects intersect strongly, or one object partially enters another's volume |

A model is "good" if it has none of these defects: smooth surfaces, no artifacts.

## Data

- `train.zip` — training data and markup.
- `test.zip` — test data.
- Downloads:
  - test: <https://disk.360.yandex.ru/d/rUSPxzoDTHK8UQ>
  - train: <https://disk.360.yandex.ru/d/CeZVSNyRGjrLUw>
  - mirror (both): <https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/AIC_data.tar>

Per object (`item_id`):

- `{item_id}.png` — 6 images of the model/scene: 4 azimuth angles at 90° steps,
  plus top and bottom views. The model is centered and fits a unit cube.
- `{item_id}.npz` — 3D model arrays `vertices` (n, 3) and `faces`.
- Markup table (`train.csv`): per `item_id`, binary values (0/1) for each defect
  and for `quality`.

## Submission

`submission.csv` with the same fields as `train.csv`: `item_id`, the 10 defect
columns, `quality`.

## Scoring (max 25 points)

1. **Prediction — 20 points.** Coefficients `w1 = 10` for the quality F1 and
   `w2 = 10` for the class-weighted artifact F1.
2. **Presentation (`presentation.pdf`) — 3 points.** 3 = interesting data
   findings, in-depth case analysis, full explanation with experiment logs;
   2 = complete approach with logs, metrics, hyperparameter research;
   1 = clear description without development artifacts; 0 = missing/gross errors.
3. **Interpretation notebook — 2 points.** 2 = several visualization methods
   analyzed, failure cases and good examples from the test set shown;
   1 = runs and shows a qualitative explanation for N test objects; 0 = missing
   or not running.

**Note:** the main-stage results for this task do **not** count toward the winner
of the competition after the final and online defense. The task continues in the
final stage with increased complexity.

## Reproducibility requirements

- The solution must run in Google Colab fully automatically, without manual code
  edits: library installs and data download start from the notebook.
- Solution files the notebook needs but does not contain (code, trained weights,
  small caches) may be hosted externally and downloaded by the notebook: a public
  link is fine as long as it stays accessible during the check and every step
  still runs from the notebook without manual preparation (organizer
  clarification, 17 Sep 2026).
- Notebook inference must reproduce exactly the predictions submitted to the
  platform (leaderboard match). Minor floating-point differences are acceptable.
- The solution must set a `random_seed`.
- Training code must also run in Colab without errors; it may exceed Colab's
  time limit, but must be fully functional from start to finish.
- Solutions violating reproducibility get 0 points for the stage.

## Solution selection

Each participant marks one uploaded solution as the final/best one for expert
evaluation. If several are marked, the last marked final solution is evaluated;
if none are marked, the last uploaded solution is evaluated.
