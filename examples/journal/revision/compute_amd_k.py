"""Compute d_amd uniqueness and novelty matrices for a custom AMD k."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from xtalmet.constants import HF_VERSION
from xtalmet.distance import distance_matrix

JOURNAL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = JOURNAL_DIR / "results" / "mp20"
TRAIN_SAMPLES_PATH = JOURNAL_DIR / "preprocess" / "train_samples.pkl.gz"


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


def load_generated_crystals(model: str) -> list:
	"""Load generated crystals for a model from Hugging Face."""
	path = hf_hub_download(
		repo_id="masahiro-negishi/xtalmet",
		filename=f"mp20/model/{model}.pkl.gz",
		repo_type="dataset",
		revision=HF_VERSION,
	)
	return load_gz(path)


def compute_amd_k(model: str, k: int, overwrite: bool = False) -> None:
	"""Compute and save d_amd embeddings and matrices using a custom k.

	Args:
		model (str): Model name, or "test".
		k (int): Number of nearest neighbors for AMD embeddings.
		overwrite (bool): Whether to overwrite existing suffixed outputs.
	"""
	dir_save = RESULTS_DIR / model
	path_emb = dir_save / f"emb_amd_{k}.pkl.gz"
	path_uni = dir_save / f"mtx_uni_amd_{k}.pkl.gz"
	path_nov = dir_save / f"mtx_nov_amd_{k}.pkl.gz"

	needs_emb = overwrite or not path_emb.exists()
	needs_uni = overwrite or not path_uni.exists()
	needs_nov = overwrite or not path_nov.exists()

	if not (needs_emb or needs_uni or needs_nov):
		print(f"All AMD k={k} outputs already exist for {model}.")
		return

	gen_embs = None

	if needs_emb or needs_uni:
		print(f"Loading generated crystals for {model}...")
		gen_xtals = load_generated_crystals(model)
		print(f"Computing uniqueness matrix for {model} with AMD k={k}...")
		mtx_uni, gen_embs, times = distance_matrix(
			distance="amd",
			xtals_1=gen_xtals,
			xtals_2=None,
			normalize=True,
			multiprocessing=False,
			verbose=True,
			args_emb={"k": k},
		)
		if needs_emb:
			save_gz(gen_embs, path_emb)
			print(f"Saved {path_emb}")
		if needs_uni:
			save_gz(mtx_uni, path_uni)
			print(f"Saved {path_uni}")
		print(
			"Uniqueness timing: "
			f"embedding={times['emb_1']:.2f}s, matrix={times['d_mtx']:.2f}s"
		)
	elif needs_nov:
		print(f"Loading generated AMD k={k} embeddings for {model}...")
		gen_embs = load_gz(path_emb)

	if needs_nov:
		if not TRAIN_SAMPLES_PATH.exists():
			raise FileNotFoundError(
				f"MP20 train crystals not found: {TRAIN_SAMPLES_PATH}. "
				"Run examples/journal/visualize.ipynb preprocessing first."
			)
		print("Loading MP20 train crystals...")
		train_xtals = load_gz(TRAIN_SAMPLES_PATH)
		print(f"Computing novelty matrix for {model} with AMD k={k}...")
		mtx_nov, _, _, times = distance_matrix(
			distance="amd",
			xtals_1=gen_embs,
			xtals_2=train_xtals,
			normalize=True,
			multiprocessing=False,
			verbose=True,
			args_emb={"k": k},
		)
		save_gz(mtx_nov, path_nov)
		print(f"Saved {path_nov}")
		print(
			"Novelty timing: "
			f"train_embedding={times['emb_2']:.2f}s, matrix={times['d_mtx']:.2f}s"
		)


def main() -> None:
	"""Run the command-line interface."""
	parser = argparse.ArgumentParser(
		description="Compute d_amd uniqueness and novelty matrices for custom k."
	)
	parser.add_argument(
		"--model",
		type=str,
		required=True,
		help='Model name, or "test".',
	)
	parser.add_argument(
		"--k",
		type=positive_int,
		required=True,
		help="AMD embedding length / number of nearest neighbors.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing suffixed output files.",
	)
	args = parser.parse_args()
	compute_amd_k(**vars(args))


if __name__ == "__main__":
	main()
