#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

KS=(25 50 200)
mapfile -t MODELS < <(
	uv run python -c \
		'from examples.journal.revision.constant import MODELS_TEST; print("\n".join(MODELS_TEST))'
)

for model in "${MODELS[@]}"; do
	for k in "${KS[@]}"; do
		printf 'Computing AMD k=%s for model=%s\n' "${k}" "${model}"
		uv run python examples/journal/revision/compute_amd_k.py \
			--model "${model}" \
			--k "${k}"
	done
done
