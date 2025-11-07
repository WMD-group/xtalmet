"""Test screening functions."""

import datetime
import gzip
import os
import pickle

import pytest

from xtalmet.crystal import Crystal
from xtalmet.stability import compute_ehull, compute_stability_scores


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


@pytest.mark.parametrize(
	"diagram, mace_model",
	[("mp_250618", "mace-mh-1"), ("mp_250618", "medium-mpa-0"), ("mp", "mace-mh-1")],
)
def test_compute_ehull(
	tmpdir: str, prepare_gen_xtals: list[Crystal], diagram: str, mace_model: str
):
	"""Test compute_ehull."""
	gen_xtals = prepare_gen_xtals

	if diagram == "mp" and os.getenv("MP_API_KEY") is None:
		pytest.skip("MP_API_KEY is not set.")

	ehull_1 = compute_ehull(gen_xtals, diagram, mace_model)
	ehull_2 = compute_ehull(gen_xtals, diagram, mace_model, tmpdir)
	assert all(ehull_1 == ehull_2)
	assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
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


@pytest.mark.parametrize(
	"diagram, mace_model, binary, kwargs",
	[
		("mp_250618", "mace-mh-1", True, {"threshold": 0.2}),
		("mp_250618", "medium-mpa-0", False, {"intercept": 0.4}),
	],
)
def test_compute_stability_scores(
	tmpdir: str,
	prepare_gen_xtals: list[Crystal],
	diagram: str,
	mace_model: str,
	binary: bool,
	kwargs: dict,
):
	"""Test compute_stability_scores."""
	gen_xtals = prepare_gen_xtals

	scores_1 = compute_stability_scores(
		gen_xtals, diagram, mace_model, dir_intermediate=tmpdir, binary=binary, **kwargs
	)
	scores_2 = compute_stability_scores(
		gen_xtals, diagram, mace_model, dir_intermediate=tmpdir, binary=binary, **kwargs
	)
	assert all(scores_1 == scores_2)
	assert all(scores_1 >= 0) and all(scores_1 <= 1)
