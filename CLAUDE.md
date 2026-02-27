# xtalmet

Crystal structure evaluation package for generative models (Validity, Stability, Uniqueness, Novelty — VSUN metrics).

## Build & Environment

- **Package manager**: `uv` (not pip). All commands go through `uv run` or `uv add`.
- **Python version**: 3.12+ (see `.python-version`)

## Testing

```bash
uv run pytest tests/ -s # all tests
```

- Test data (POSCAR files, pickles) lives in `tests/data/`
- `test_stability.py` is slow — it loads MACE ML force fields; skip unless testing stability changes

## Code Style

- **Indentation**: tabs (not spaces) — enforced by Ruff formatter
- **Line length**: 88 characters
- **Quotes**: double quotes
- **Docstrings**: Google-style
- **Formatter/linter**: Ruff — run with `uv run ruff format .` and `uv run ruff check .`

## Architecture

Six modules in `src/xtalmet/`:

| Module | Purpose |
|---|---|
| `crystal.py` | `Crystal` class extending `pymatgen.Structure` |
| `distance.py` | All distance functions + embedding helpers |
| `evaluator.py` | `Evaluator` — main user-facing API for VSUN evaluation |
| `validity.py` | `SMACTValidator`, `StructureValidator` |
| `stability.py` | `StabilityCalculator` (MACE energy calculation + MP phase diagram) |
| `constants.py` | Type aliases and supported metric lists |

**Distance metric types** (defined in `constants.py`):
- Binary (0.0 or 1.0): `smat`, `comp`, `wyckoff`
- Continuous normalized (0–1): `elmd+amd`
- Continuous unnormalized (can be normalized): `magpie`, `pdd`, `amd`, `elmd`

## Documentation

```bash
cd docs && uv run make clean && uv run python build_docs.py # build Sphinx HTML
```

Docs deploy automatically to GitHub Pages on push to `main` via `.github/workflows/sphinx.yml`.

## Common Gotchas

- Heavy dependencies (PyTorch, MACE) — imports can be slow on first load
- `distance.py` uses multiprocessing for matrix computations; set `n_jobs` appropriately
- Embeddings can be pre-computed and cached (see `Evaluator` API) to avoid recomputation
- Reference dataset (MP20) is downloaded from Hugging Face on first use
