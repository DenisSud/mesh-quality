"""Shared CLI scaffold for competition tasks.

Every task entry point (``main/<task>/main.py``) builds its parser through
:class:`TaskCLI`, so the command surface is identical across tasks:

- ``train``     -- fit the model(s) and save artifacts under the task directory
- ``predict``   -- run inference and write the submission file
- ``score``     -- evaluate a submission with the task metric
- ``visualize`` -- produce an interpretation view for one item

Task-specific commands (e.g. QAOA target generation) are added with
:meth:`TaskCLI.add_command` and get the common options too.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

Handler = Callable[[argparse.Namespace], None]

STANDARD_COMMANDS: dict[str, str] = {
    "train": "fit the model(s) and save artifacts under the task directory",
    "predict": "run inference and write the submission file",
    "score": "evaluate a submission with the task metric",
    "visualize": "produce an interpretation view for one item",
}


class TaskCLI:
    """Argparse scaffold with the shared command surface.

    ``handlers`` maps standard command names to callables from the task's
    ``solution`` module; commands are registered in the fixed
    :data:`STANDARD_COMMANDS` order and only those provided are added.
    """

    def __init__(
        self,
        prog: str,
        description: str,
        task_dir: str | Path,
        handlers: Mapping[str, Handler] | None = None,
    ):
        self.task_dir = Path(task_dir)
        self.parser = argparse.ArgumentParser(prog=prog, description=description)
        self._sub = self.parser.add_subparsers(dest="command", required=True)
        #: command name -> its subparser, so tasks can add task-specific options
        self.parsers: dict[str, argparse.ArgumentParser] = {}
        handlers = handlers or {}
        unknown = set(handlers) - set(STANDARD_COMMANDS)
        if unknown:
            raise ValueError(f"unknown standard commands: {sorted(unknown)}")
        for name, help_text in STANDARD_COMMANDS.items():
            if name in handlers:
                self._add(name, help_text, handlers[name])

    def add_command(self, name: str, help_text: str, handler: Handler) -> argparse.ArgumentParser:
        """Register a task-specific command with the common options."""
        return self._add(name, help_text, handler)

    def _add(self, name: str, help_text: str, handler: Handler) -> argparse.ArgumentParser:
        parser = self._sub.add_parser(name, help=help_text, description=help_text)
        parser.set_defaults(func=handler)
        self.parsers[name] = parser
        parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
        parser.add_argument(
            "--data-dir",
            type=Path,
            default=self.task_dir / "data",
            help="data directory (default: <task-dir>/data)",
        )
        if name in ("predict", "score"):
            parser.add_argument(
                "--submission",
                type=Path,
                default=self.task_dir / "submission.csv",
                help="submission path (default: <task-dir>/submission.csv)",
            )
        elif name == "visualize":
            parser.add_argument("item_id", help="item to visualize (id from the test set)")
            parser.add_argument(
                "--out-dir",
                type=Path,
                default=self.task_dir / "visualizations",
                help="output directory (default: <task-dir>/visualizations)",
            )
        return parser

    def run(self, argv: list[str] | None = None) -> int:
        args = self.parser.parse_args(argv)
        args.task_dir = self.task_dir
        args.func(args)
        return 0
