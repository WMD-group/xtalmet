"""Test screening functions."""

import gzip
import os
import pickle

import pytest

from xtalmet.crystal import Crystal
from xtalmet.validity import screen_smact


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


def test_screen_smact(tmpdir: str, prepare_gen_xtals: list[Crystal]):
	"""Test screen_smact."""
	gen_xtals = prepare_gen_xtals

	screened_1 = screen_smact(gen_xtals)
	screened_2 = screen_smact(gen_xtals, tmpdir)
	assert all(screened_1 == screened_2)
	assert os.path.exists(os.path.join(tmpdir, "valid_smact.pkl.gz"))
