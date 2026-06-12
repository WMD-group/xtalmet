"""Compute sampled model-pair d_elmd+amd distance matrices."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from constant import MODELS_TEST, PREPROCESS_DIR, RESULTS_DIR

from xtalmet.distance import distance_matrix

DISTANCE = "elmd+amd"
DEFAULT_OUTPUT_DIR = PREPROCESS_DIR / "model_pair_elmd_amd"


def load_gz(path: str | Path) -> Any:
	"""Load a gzip-pickled object."""
	with gzip.open(path, "rb") as f:
		return pickle.load(f)


def save_gz(data: Any, path: str | Path) -> None:
	"""Save a gzip-pickled object."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with gzip.open(path, "wb") as f:
		pickle.dump(data, f)


def positive_int(value: str) -> int:
	"""Parse a positive integer for argparse."""
	try:
		parsed = int(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("must be an integer") from exc
	if parsed <= 0:
		raise argparse.ArgumentTypeError("must be positive")
	return parsed


def non_negative_int(value: str) -> int:
	"""Parse a non-negative integer for argparse."""
	try:
		parsed = int(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("must be an integer") from exc
	if parsed < 0:
		raise argparse.ArgumentTypeError("must be non-negative")
	return parsed


def result_dir(output_dir: str | Path, sample_size: int, seed: int) -> Path:
	"""Return the output directory for one sampling configuration."""
	return Path(output_dir) / f"n{sample_size}_seed{seed}"


def indices_path(output_dir: str | Path, sample_size: int, seed: int) -> Path:
	"""Return the sampled-indices metadata path."""
	return result_dir(output_dir, sample_size, seed) / "sample_indices.json"


def matrix_path(
	output_dir: str | Path,
	sample_size: int,
	seed: int,
	model_1: str,
	model_2: str,
) -> Path:
	"""Return the cached matrix path for an ordered model pair."""
	return (
		result_dir(output_dir, sample_size, seed) / f"mtx_{model_1}__{model_2}.pkl.gz"
	)


def load_embeddings(model: str) -> list:
	"""Load cached d_elmd+amd embeddings for a model."""
	path = RESULTS_DIR / model / "emb_elmd+amd.pkl.gz"
	if not path.exists():
		raise FileNotFoundError(
			f"Missing cached embeddings for {model}: {path}. "
			"Run examples/journal/eval.py for d_elmd+amd first."
		)
	return load_gz(path)


def load_or_sample_indices(
	models: list[str],
	embeddings: dict[str, list],
	sample_size: int,
	seed: int,
	output_dir: str | Path,
	overwrite: bool,
) -> dict[str, list[int]]:
	"""Load existing sample indices or create reproducible random samples."""
	path = indices_path(output_dir, sample_size, seed)
	if path.exists() and not overwrite:
		with open(path) as f:
			payload = json.load(f)
		if payload["models"] != models:
			raise ValueError(
				f"Existing sample indices use models {payload['models']}, "
				f"but requested {models}."
			)
		return {model: payload["indices"][model] for model in models}

	rng = np.random.default_rng(seed)
	indices = {}
	for model in models:
		n_available = len(embeddings[model])
		if n_available < sample_size:
			raise ValueError(
				f"{model} has only {n_available} samples; requested {sample_size}."
			)
		indices[model] = np.sort(
			rng.choice(n_available, size=sample_size, replace=False)
		).tolist()

	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w") as f:
		json.dump(
			{
				"distance": DISTANCE,
				"sample_size": sample_size,
				"seed": seed,
				"models": models,
				"indices": indices,
			},
			f,
			indent=2,
			sort_keys=True,
		)
	print(f"Saved {path}")
	return indices


def compute_model_pair_matrices(
	models: list[str],
	sample_size: int,
	seed: int,
	output_dir: str | Path = DEFAULT_OUTPUT_DIR,
	overwrite: bool = False,
	multiprocessing: bool = True,
	n_processes: int | None = None,
) -> None:
	"""Compute sampled d_elmd+amd distance matrices for all unordered model pairs."""
	print("Loading embeddings...")
	embeddings = {model: load_embeddings(model) for model in models}
	indices = load_or_sample_indices(
		models=models,
		embeddings=embeddings,
		sample_size=sample_size,
		seed=seed,
		output_dir=output_dir,
		overwrite=overwrite,
	)
	samples = {
		model: [embeddings[model][idx] for idx in indices[model]] for model in models
	}

	for i, model_1 in enumerate(models):
		for model_2 in models[i:]:
			path = matrix_path(output_dir, sample_size, seed, model_1, model_2)
			if path.exists() and not overwrite:
				print(f"Skipping existing {path}")
				continue

			print(f"Computing {model_1} vs {model_2}...")
			start = time.time()
			mtx = distance_matrix(
				distance=DISTANCE,
				xtals_1=samples[model_1],
				xtals_2=samples[model_2],
				normalize=True,
				multiprocessing=multiprocessing,
				n_processes=n_processes,
			)
			save_gz(mtx, path)
			elapsed = time.time() - start
			print(f"Saved {path} ({elapsed:.2f}s)")


def main() -> None:
	"""Run the command-line interface."""
	parser = argparse.ArgumentParser(
		description="Compute sampled model-pair d_elmd+amd distance matrices."
	)
	parser.add_argument(
		"--sample-size",
		type=positive_int,
		default=1000,
		help="Number of samples to draw per model.",
	)
	parser.add_argument(
		"--seed",
		type=non_negative_int,
		default=0,
		help="Random seed for reproducible sampling.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory for sampled indices and pair matrices.",
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=MODELS_TEST,
		choices=MODELS_TEST,
		help="Models to include. Defaults to MODELS_TEST.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing indices and pair matrices for this configuration.",
	)
	parser.add_argument(
		"--multiprocessing",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Use multiprocessing for the ElMD part of the distance computation.",
	)
	parser.add_argument(
		"--n-processes",
		type=positive_int,
		default=None,
		help="Maximum number of worker processes.",
	)
	args = parser.parse_args()
	compute_model_pair_matrices(**vars(args))


if __name__ == "__main__":
	main()
