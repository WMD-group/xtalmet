"""Audit published MP20 embeddings against current defaults without replacing data."""

import argparse
import csv
import functools
import gzip
import hashlib
import json
import multiprocessing
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from pymatgen.io.cif import CifParser

from xtalmet._cache import MP20_REFERENCE_REVISION, _software_identity
from xtalmet.constants import SUPPORTED_DISTANCES
from xtalmet.crystal import Crystal
from xtalmet.distance import _compute_embedding_worker, _embedding_is_valid

SOURCE_REVISION = "73874c48a46b74fedaa161fdac512b0a75e2d336"


def compare_embedding(distance: str, pair: tuple) -> tuple[bool, float]:
	"""Compare one embedding at fixed tolerances, including failure state and shape."""
	structure, expected = pair
	actual = _compute_embedding_worker(structure, distance)
	if _embedding_is_valid(distance, actual) != _embedding_is_valid(distance, expected):
		return False, float("inf")
	if not _embedding_is_valid(distance, actual):
		return True, 0.0
	if distance == "smat":
		if actual.species != expected.species:
			return False, float("inf")
		fractional_delta = actual.frac_coords - expected.frac_coords
		fractional_delta -= np.rint(fractional_delta)
		lattice_error = np.max(np.abs(actual.lattice.matrix - expected.lattice.matrix))
		matches = np.allclose(
			actual.lattice.matrix, expected.lattice.matrix, rtol=1e-8, atol=1e-10
		) and np.allclose(fractional_delta, 0, rtol=1e-8, atol=1e-10)
		return bool(matches), float(
			max(lattice_error, np.max(np.abs(fractional_delta @ actual.lattice.matrix)))
		)
	if distance in ("comp", "wyckoff", "elmd"):
		return actual == expected, 0.0
	if distance == "elmd+amd":
		if actual[0] != expected[0]:
			return False, float("inf")
		actual, expected = actual[1], expected[1]
	actual, expected = np.asarray(actual), np.asarray(expected)
	if actual.shape != expected.shape:
		return False, float("inf")
	return (
		bool(np.allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True)),
		float(np.max(np.abs(actual - expected))),
	)


def verify(source_csv: Path, report_path: Path, n_processes: int = 2) -> None:
	"""Verify source order and every published embedding; write only a new report."""
	if report_path.exists():
		raise FileExistsError(report_path)
	start = time.monotonic()
	structures = []
	with source_csv.open() as handle, warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		for row in csv.DictReader(handle):
			structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[
				0
			]
			structures.append(Crystal.from_Structure(structure))
	print(f"Parsed {len(structures)} structures from source CSV.", flush=True)
	with source_csv.open("rb") as handle:
		source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
	report = {
		"source_revision": SOURCE_REVISION,
		"source_csv_sha256": source_sha256,
		"reference_revision": MP20_REFERENCE_REVISION,
		"software": _software_identity(),
		"count": len(structures),
		"rtol": 1e-8,
		"atol": 1e-10,
		"structure_comparison": "Ordered species; lattice allclose; fractional coordinates modulo integer translations. Structure errors are in angstroms.",
		"n_processes": n_processes,
		"distances": {},
	}
	# Verify raw structures first, before treating their order as authoritative.
	for distance in ["smat", *[d for d in SUPPORTED_DISTANCES if d != "smat"]]:
		path = hf_hub_download(
			"masahiro-negishi/xtalmet",
			f"mp20/train/train_{distance}.pkl.gz",
			repo_type="dataset",
			revision=MP20_REFERENCE_REVISION,
		)
		with gzip.open(path, "rb") as handle:
			expected = pickle.load(handle)
		if len(expected) != len(structures):
			raise ValueError(f"{distance}: reference count differs from source CSV")
		failed = []
		maximum = 0.0
		metric_start = time.monotonic()
		with multiprocessing.Pool(n_processes) as pool:
			results = pool.imap(
				functools.partial(compare_embedding, distance),
				zip(structures, expected, strict=True),
				chunksize=32,
			)
			for index, (matches, error) in enumerate(results):
				if not matches:
					failed.append(index)
				maximum = max(maximum, error)
				if (index + 1) % 2000 == 0:
					print(f"{distance}: {index + 1}/{len(structures)}", flush=True)
		report["distances"][distance] = {
			"mismatch_indices": failed,
			"max_absolute_error": maximum,
			"seconds": time.monotonic() - metric_start,
		}
		print(
			f"{distance}: {len(failed)} mismatches, maximum error {maximum}", flush=True
		)
		del expected
	report["seconds"] = time.monotonic() - start
	report_path.write_text(json.dumps(report, indent=2) + "\n")
	if any(item["mismatch_indices"] for item in report["distances"].values()):
		raise ValueError(f"Published embeddings differ; see {report_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source_csv", type=Path)
	parser.add_argument("report", type=Path)
	parser.add_argument("--n-processes", type=int, default=2)
	args = parser.parse_args()
	if args.n_processes < 1:
		parser.error("--n-processes must be positive")
	verify(args.source_csv, args.report, args.n_processes)
