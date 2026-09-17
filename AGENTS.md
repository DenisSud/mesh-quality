# AGENTS.md

Notes for agents and humans working on this repo — the **AI Journey 3D
mesh-quality** solution (Sber AI, generative models).

## Layout

```
main/mesh-quality/        the task: main.py CLI, src/mesh_quality/ package,
                          tests/, notebooks/, presentation/, tools/
docs/tasks/mesh-quality.md  full task spec and download links
src/aiijc/                shared task-CLI scaffold (aiijc.cli.TaskCLI)
scripts/check_gpu.py      GPU smoke test for PyTorch / JAX / cuTile
devenv.nix, pyproject.toml, uv.lock   the shared dev environment
```

## Conventions

- `main.py` only builds the CLI and dispatches; solution logic lives in
  `src/mesh_quality/`. Handlers receive resolved paths through `args`, so task
  code never depends on the current working directory.
- Run from the repo root, inside the environment:
  `devenv shell -- uv run python main/mesh-quality/main.py <command>`.
- Dependencies go into the root `pyproject.toml` (single environment). CUDA
  libraries come from pip wheels; `devenv.nix` patches their bundled ELF
  binaries for NixOS and exports the driver paths. After changing CUDA
  packages, verify with `devenv shell -- check-gpu`.
- Commit code, docs, configs and small results; keep datasets, checkpoints and
  caches out of git (see `.gitignore`).

## Task CLI

Built with `aiijc.cli.TaskCLI`: `train | predict | score | visualize` plus
task-specific `features | fuse | resubmit | logscore | deliver`.
