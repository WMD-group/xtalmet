# xtalmet agent guide

`xtalmet` is a Python package for crystal-structure distance functions and VSUN
(Validity, Stability, Uniqueness, Novelty) evaluation metrics for generative
crystal models.

## Navigation

Read only the files and sections relevant to the requested change:

- `README.md` and `docs/source/`: installation, tutorial, and API documentation.
- `src/xtalmet/`: `evaluator.py` owns VSUN evaluation; `distance.py` owns
  distances and embeddings; `crystal.py` owns the `Crystal` structure wrapper;
  `validity.py` owns validators; `stability.py` owns MACE energies and Materials
  Project phase diagrams; `constants.py` owns metric lists and type aliases.
- `tests/`: pytest suite, with fixtures under `tests/data/`.
- `examples/`: tutorials and paper/workshop evaluations.
- `dev/`: dataset preparation, sampling, and upload utilities.

## Working rules

- Start with `git status --short` and `git rev-parse HEAD`. Keep the starting
  commit for the final diff and preserve pre-existing changes outside the task.
- Make the smallest change that meets the requested behavior. Reuse local
  patterns; avoid speculative abstractions and unrelated cleanup.
- Preserve authoritative inputs, generated data, cached results, and ignored
  result directories unless changing them is part of the request.
- Follow the Ruff settings in `pyproject.toml`: tabs, double quotes, an
  88-character line length, and Google-style docstrings. Keep changed public
  APIs typed and documented.
- Update information in its existing owning document and link rather than
  duplicate it elsewhere. Keep this guide limited to durable repository rules;
  personal preferences belong in global instructions.

## Environment and commands

Use `uv` for environments and dependencies, not system Python installations.
`.python-version` selects the development interpreter (currently 3.12),
`pyproject.toml` declares supported Python versions and dependencies, and
`uv.lock` records the resolved environment. Change dependency declarations and
the lockfile together. A declared Python range is not evidence that every
version has been tested.

Run these commands from the repository root. Prefix shell commands with `rtk`
when available; use `rtk proxy` when raw output is needed.

| Purpose | Command |
|---|---|
| Set up the locked development environment | `uv sync --locked --group dev` |
| Distance tests | `uv run --locked pytest tests/test_distance.py -s` |
| Evaluator tests | `uv run --locked pytest tests/test_evaluator.py -s` |
| Full test suite | `uv run --locked pytest tests/ -s` |
| Lint changed Python files | `uv run --locked ruff check <changed-files>` |
| Check their formatting | `uv run --locked ruff format --check <changed-files>` |
| Format changed Python files | `uv run --locked ruff format <changed-files>` |
| Check packaging after build configuration changes | `uv build` |
| Install local pre-commit hooks when needed | `uv run --locked pre-commit install` |

Replace `<changed-files>` with the relevant paths. Run the most focused tests
while iterating and add regression tests for substantive behavior changes.
For Python changes, run the relevant tests and both Ruff checks. For
documentation-only edits, verify paths, commands, and consistency with the code;
exercise changed procedures safely, using a dry run where appropriate.

The full suite and some Evaluator cases load MACE models or download reference
data. `tests/test_stability.py` is expensive; run it when stability behavior or
its dependencies change. Do not treat other test files as automatically offline
or lightweight. Separate pre-existing failures, regressions, and unrun checks.
Do not claim a check passed unless it actually ran and passed.

**Docs build safety:** `uv run --locked python docs/build_docs.py` builds committed
local `main` and the configured tags in temporary clones. It does not switch the
source checkout's branch or change its inputs. All versions must build before
output is copied to `docs/build/html/`; unrelated output files are retained.
The local refs and documentation dependencies must already exist. This command
does not validate uncommitted documentation changes. The deployment workflow is
`.github/workflows/sphinx.yml`.

## Scientific and API constraints

- Preserve dense distance matrices in the public API and caches. Binary
  distances (`smat`, `comp`, `wyckoff`) use 0 for a match and 1 for a mismatch;
  failed computations must remain distinguishable from valid mismatches.
- StructureMatcher matches depend on tolerances and need not be transitive.
  Do not substitute representative-only comparisons or connected components
  for pairwise results. Preserve configured matcher behavior when pruning pairs.
- The public parallelism controls are `multiprocessing` and `n_processes`.
  Choose process counts deliberately; the PDD backend's `n_jobs` is a separate
  option. Avoid multiplying worker pools or exhausting memory in checks.
- Keep continuous distance normalization and VSUN aggregation semantics intact
  unless the requested change explicitly alters them.
- Distinguish intended behavior from observed results. Do not silently change
  datasets, tolerances, units, or evaluation conditions to make checks pass.
- For performance changes, verify result equivalence and report input size,
  settings, environment, and measurements. Passing software tests alone does
  not establish scientific validity or superiority to a baseline.

## Review and handoff

1. Inspect the final diff for correctness, scope, API compatibility, and
   scientific consequences. Include staged changes, relevant untracked files,
   and any task commits relative to the recorded starting commit.
2. Run the applicable checks above. Required failing or unrun checks prevent
   claiming full completion; report the failure or limitation explicitly.
3. For code, dependency, configuration, CI, or execution-procedure changes,
   delegate an independent review to another agent using
   `ponytail:ponytail-review`. Include the task scope, relevant diff, new files,
   and surrounding code. Changes to procedures documented in Markdown also
   require this review; prose-only edits are exempt. If the skill or independent
   agent is unavailable, report the blocker rather than substituting self-review.
4. Resolve findings with a fix or an explained reason for not adopting them.
   After fixes, rerun affected checks and repeat the independent review of the
   changed portions. Review does not replace tests or scientific validation.
5. Update affected user-facing documentation. Run `git diff --check` and
   `git diff --cached --check`; include `git diff --check <start-commit> HEAD`
   if the task contains commits. Recheck the final status and diff after these
   edits. Report changes, checks, review outcomes, and remaining limitations.

Commit and push only as authorized by the user. A clean working tree is not a
completion requirement; unrelated user work must remain intact.
