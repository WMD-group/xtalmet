"""Test screening functions."""

import gzip
import os
import pickle

import pytest

from xtalmet.crystal import Crystal
from xtalmet.validity import validity_smact, validity_structure


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


def test_validity_smact(tmpdir: str, prepare_gen_xtals: list[Crystal]):
	"""Test validity_smact."""
	gen_xtals = prepare_gen_xtals

	screened_1 = validity_smact(gen_xtals)
	screened_2 = validity_smact(gen_xtals, tmpdir)
	assert all(screened_1 == screened_2)
	assert os.path.exists(os.path.join(tmpdir, "valid_smact.pkl.gz"))


@pytest.mark.parametrize(
	"threshold_distance, threshold_volume",
	[
		(0.5, 0.1),
		(0.6, 0.2),
	],
)
def test_validity_structure(
	tmpdir: str,
	prepare_gen_xtals: list[Crystal],
	threshold_distance: float,
	threshold_volume: float,
):
	"""Test validity_structure."""
	gen_xtals = prepare_gen_xtals

	screened_1 = validity_structure(
		gen_xtals, None, threshold_distance, threshold_volume
	)
	screened_2 = validity_structure(
		gen_xtals, tmpdir, threshold_distance, threshold_volume
	)
	assert all(screened_1 == screened_2)
	assert os.path.exists(os.path.join(tmpdir, "valid_structure.pkl.gz"))
