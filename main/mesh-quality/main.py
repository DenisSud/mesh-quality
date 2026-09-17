#!/usr/bin/env python
"""CLI index for the 3D Mesh Quality Control task.

Run from the repo root:
    devenv shell -- uv run python main/mesh-quality/main.py train
    devenv shell -- uv run python main/mesh-quality/main.py predict
    devenv shell -- uv run python main/mesh-quality/main.py score
    devenv shell -- uv run python main/mesh-quality/main.py visualize <item_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR / "src"))

from aiijc.cli import TaskCLI  # noqa: E402
from mesh_quality import images, model, solution  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cli = TaskCLI(
        prog="mesh-quality",
        description="Defect and quality classification for 3D meshes (main stage).",
        task_dir=TASK_DIR,
        handlers={
            "train": solution.train,
            "predict": solution.predict,
            "score": solution.score,
            "visualize": solution.visualize,
        },
    )
    feat = cli.add_command(
        "features",
        "cache frozen DINOv3 render features under cache/",
        solution.features,
    )
    feat.add_argument("--model", default="s", choices=sorted(images.MODELS))
    feat.add_argument("--split", default="all", choices=["train", "test", "all"])
    feat.add_argument("--limit", type=int, default=None, help="first N items only (debug)")

    tr = cli.parsers["train"]
    probe_cfg, recipe = model.ProbeConfig, model.TrainRecipe
    tr.add_argument(
        "--model",
        default="s",
        choices=sorted([*images.MODELS, "geometry"]),
        help="DINOv3 feature cache (or 'geometry' for the HistGB fallback)",
    )
    tr.add_argument("--d", type=int, default=probe_cfg.d, help=f"probe width (default: {probe_cfg.d})")
    tr.add_argument("--depth", type=int, default=probe_cfg.depth, help=f"probe transformer blocks (default: {probe_cfg.depth})")
    tr.add_argument("--heads", type=int, default=probe_cfg.heads, help=f"attention heads (default: {probe_cfg.heads})")
    tr.add_argument("--dropout", type=float, default=probe_cfg.dropout)
    tr.add_argument("--folds", type=int, default=recipe.folds)
    tr.add_argument("--epochs", type=int, default=recipe.epochs)
    tr.add_argument("--batch", type=int, default=recipe.batch)
    tr.add_argument("--lr", type=float, default=recipe.lr)
    tr.add_argument("--wd", type=float, default=recipe.wd)
    tr.add_argument("--full-seeds", type=int, default=recipe.full_seeds, help="full-data refits to ensemble")
    tr.add_argument("--limit", type=int, default=None, help="first N items only (debug)")
    tr.add_argument("--device", default=None)

    pr = cli.parsers["predict"]
    pr.add_argument("--model", default="s", choices=sorted([*images.MODELS, "geometry"]))
    pr.add_argument(
        "--models",
        default=None,
        help="comma-separated probe keys; two or more are averaged as a fusion, "
        "using the thresholds recorded by the `fuse` command",
    )
    pr.add_argument("--device", default=None)

    vi = cli.parsers["visualize"]
    vi.add_argument("--model", default="s", choices=sorted(images.MODELS))
    vi.add_argument("--split", default="test", choices=["train", "test"])
    vi.add_argument("--labels", default=None, help="comma-separated labels (default: predicted positives)")
    vi.add_argument("--device", default=None)

    rs = cli.add_command(
        "resubmit",
        "re-threshold a saved test-probability matrix into a submission (no model)",
        solution.resubmit,
    )
    rs.add_argument("--probs", type=Path, required=True, help="cache/probs_test_<model>.npy")
    rs.add_argument("--thresholds", required=True, help="10 comma-separated values, label order")
    rs.add_argument("--model", default="custom", help="label recorded in submissions.csv")
    rs.add_argument("--notes", default="", help="free-form note recorded in submissions.csv")
    rs.add_argument("--submission", type=Path, default=TASK_DIR / "submission.csv")

    fu = cli.add_command("fuse", "average several probes' test probabilities", solution.fuse)
    fu.add_argument("--models", required=True, help="comma-separated keys, e.g. s,b")
    fu.add_argument("--weights", default=None, help="comma-separated weights (default: equal)")

    ls = cli.add_command(
        "logscore",
        "record the leaderboard score of a logged submission",
        solution.logscore,
    )
    ls.add_argument("--id", required=True, help="submission id from submissions.csv, e.g. s02")
    ls.add_argument("--score", type=float, default=None)
    ls.add_argument("--place", type=int, default=None)
    ls.add_argument("--notes", default="")

    dv = cli.add_command(
        "deliver",
        "stage the Colab reproduction bundle (code + probe + references) as a zip",
        solution.deliver,
    )
    dv.add_argument("--models", default="s", help="comma-separated probe keys to ship, e.g. s,b")
    dv.add_argument("--ref-model", default=None, help="probe key that reproduces submission.csv")
    dv.add_argument("--out", type=Path, default=TASK_DIR / "deliver")
    dv.add_argument("--submission", type=Path, default=TASK_DIR / "submission.csv")
    dv.add_argument("--cache", action="store_true", help="also copy dinos_test.npz next to the zip")
    dv.add_argument("--train-cache", action="store_true", help="also copy dinos_train.npz (2.6 GB)")
    dv.add_argument("--ref-fuse3", action="store_true",
                    help="reference = fuse3 (s+b+L4) fusion; ships mesh tokens + refit ckpt + tools")
    return cli.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
