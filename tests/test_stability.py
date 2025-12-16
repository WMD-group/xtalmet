"""Test StabilityCalculator."""

import gzip
import os
import pickle

import pytest

from xtalmet.crystal import Crystal
from xtalmet.stability import StabilityCalculator


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


class TestStabilityCalculator:
	"""Test StabilityCalculator class."""

	@pytest.mark.parametrize(
		"diagram, mace_model, binary, threshold, intercept",
		[
			("mp_250618", "mh-1", True, 0.1, None),
			("mp_250618", "medium-mpa-0", False, None, 0.4289),
			("mp", "mh-1", True, 0.2, None),
		],
	)
	def test_init(
		self,
		diagram: str,
		mace_model: str,
		binary: bool,
		threshold: float,
		intercept: float,
	):
		"""Test initialization of StabilityCalculator."""
		if diagram == "mp" and os.getenv("MP_API_KEY") is None:
			pytest.skip("MP_API_KEY is not set.")
		calculator = StabilityCalculator(
			diagram=diagram,
			mace_model=mace_model,
			binary=binary,
			threshold=threshold,
			intercept=intercept,
		)
		assert isinstance(calculator, StabilityCalculator)

	@pytest.mark.parametrize(
		"diagram, mace_model",
		[
			("mp_250618", "mh-1"),
			("mp_250618", "medium-mpa-0"),
		],
	)
	def test_ehull(
		self, prepare_gen_xtals: list[Crystal], diagram: str, mace_model: str
	):
		"""Test ehull calculation."""
		gen_xtals = prepare_gen_xtals
		calculator = StabilityCalculator(diagram=diagram, mace_model=mace_model)
		ehulls = calculator._ehull(gen_xtals)
		assert len(ehulls) == len(gen_xtals)
		assert all(isinstance(ehull, float) for ehull in ehulls)

	@pytest.mark.parametrize(
		"diagram, mace_model, binary, threshold, intercept",
		[
			("mp_250618", "mh-1", True, 0.1, None),
			("mp_250618", "medium-mpa-0", False, None, 0.4289),
		],
	)
	def test_compute_stability_scores(
		self,
		prepare_gen_xtals: list[Crystal],
		diagram: str,
		mace_model: str,
		binary: bool,
		threshold: float,
		intercept: float,
	):
		"""Test compute_stability_scores method."""
		gen_xtals = prepare_gen_xtals
		calculator = StabilityCalculator(
			diagram, mace_model, binary, threshold, intercept
		)
		scores_1, e_above_hulls_1, time_1 = calculator.compute_stability_scores(
			xtals=gen_xtals
		)
		scores_2, e_above_hulls_2, time_2 = calculator.compute_stability_scores(
			xtals=gen_xtals, e_above_hulls_precomputed=e_above_hulls_1
		)
		assert all(scores_1 == scores_2)
		assert all(e_above_hulls_1 == e_above_hulls_2)
		assert len(scores_1) == len(gen_xtals)
		assert all(isinstance(score, float) for score in scores_1)
		if binary:
			assert all(score in (0.0, 1.0) for score in scores_1)
		else:
			assert all(0.0 <= score <= 1.0 for score in scores_1)
		assert time_1 >= 0.0
		assert time_2 >= 0.0
