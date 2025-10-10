"""Test screening functions."""

import datetime
import gzip
import os
import pickle

import pytest

from xtalmet.crystal import Crystal
from xtalmet.screen import screen_ehull, screen_smact


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
	assert os.path.exists(os.path.join(tmpdir, "screen_smact.pkl.gz"))


@pytest.mark.parametrize(
	"diagram",
	["mp_250618", "mp"],
)
def test_screen_ehull(tmpdir: str, prepare_gen_xtals: list[Crystal], diagram: str):
	"""Test screen_ehull."""
	gen_xtals = prepare_gen_xtals

	if diagram == "mp" and os.getenv("MP_API_KEY") is None:
		pytest.skip("MP_API_KEY is not set.")

	screened_1 = screen_ehull(gen_xtals, diagram)
	screened_2 = screen_ehull(gen_xtals, diagram, tmpdir)
	assert all(screened_1 == screened_2)
	assert os.path.exists(os.path.join(tmpdir, "screen_ehull.pkl.gz"))
	if diagram == "mp":
		now = datetime.datetime.now()
		year = str(now.year)[-2:]
		month = f"{now.month:02d}"
		day = f"{now.day:02d}"
		assert os.path.exists(
			os.path.join(
				tmpdir,
				f"ppd-mp_all_entries_uncorrected_{year}{month}{day}.pkl.gz",
			)
		)
