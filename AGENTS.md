# xtalmet agent guide

`xtalmet` is a Python package for crystal-structure distance functions and VSUN
(Validity, Stability, Uniqueness, Novelty) evaluation metrics for generative
crystal models.

## Environment

- Target Python 3.12; see `.python-version`.
- Use `uv` for dependency and environment management.
- Prefer `rtk` as a command prefix when it is available; otherwise run the same
  command without `rtk`.

```bash
uv sync --group dev
```

## Common commands

```bash
# Full test suite
uv run pytest tests/ -s

# Focused tests
uv run pytest tests/test_distance.py -s
uv run pytest tests/test_evaluator.py -s

# Lint and format
uv run ruff check .
uv run ruff format .

# Build docs
cd docs && uv run python build_docs.py
```

Run the most focused relevant tests before finishing code changes. Run Ruff when
changing imports, formatting, or public Python code.

## Repository layout

- `src/xtalmet/`: package source.
- `tests/`: pytest suite and test data under `tests/data/`.
- `docs/`: Sphinx docs and `build_docs.py`.
- `examples/`: tutorial and paper/workshop examples.
- `dev/`: development utilities for datasets, samples, and uploads.

## Code style

- Ruff is the formatter and linter.
- Use tabs for indentation, double quotes, 88-character line length, and
  Google-style docstrings.
- Keep public APIs typed and documented when behavior changes.
- Prefer local project patterns over new abstractions.

## Domain notes

- `Evaluator` is the main user-facing VSUN evaluation API.
- `distance.py` owns distance metrics and embedding helpers.
- `stability.py` uses MACE and Materials Project phase-diagram logic.
- Binary metrics include `smat`, `comp`, and `wyckoff`.
- Continuous metrics include `elmd+amd`, `magpie`, `pdd`, `amd`, and `elmd`.

## Gotchas

- `tests/test_stability.py` is slow because it loads MACE ML force fields; run it
  when touching stability behavior, not for unrelated changes.
- First use can download heavy models or reference datasets from Hugging Face.
- Distance matrix work can use multiprocessing; choose `n_jobs` deliberately.
- Preserve generated data and ignored result directories unless the task
  explicitly asks to change them.

## Done criteria

- Keep changes scoped to the request.
- Add or update focused tests for behavior changes when practical.
- Run relevant checks, then review the diff before final response.
- Report checks run and any checks skipped.
