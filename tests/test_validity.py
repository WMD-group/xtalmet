"""Test validators."""

import gzip
import pickle

import numpy as np
import pytest

from xtalmet.crystal import Crystal
from xtalmet.validity import SMACTValidator, StructureValidator, Validator


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


def test_SMACTValidator(prepare_gen_xtals: list[Crystal]):
	"""Test SMACTValidator."""
	gen_xtals = prepare_gen_xtals

	validator = SMACTValidator()
	scores = validator.validate(xtals=gen_xtals)
	assert len(scores) == len(gen_xtals)
	assert all(0.0 <= score <= 1.0 for score in scores)
	assert all(isinstance(score, float) for score in scores)


@pytest.mark.parametrize(
	"threshold_distance, threshold_volume",
	[
		(0.5, 0.1),
		(0.6, 0.2),
		(None, None),
	],
)
def test_StructureValidator(
	prepare_gen_xtals: list[Crystal],
	threshold_distance: float,
	threshold_volume: float,
):
	"""Test StructureValidator."""
	gen_xtals = prepare_gen_xtals

	if threshold_distance is None and threshold_volume is None:
		validator = StructureValidator()
	else:
		validator = StructureValidator(
			threshold_distance=threshold_distance, threshold_volume=threshold_volume
		)
	scores = validator.validate(xtals=gen_xtals)
	assert len(scores) == len(gen_xtals)
	assert all(0.0 <= score <= 1.0 for score in scores)
	assert all(isinstance(score, float) for score in scores)


@pytest.mark.parametrize(
	"methods, kwargs",
	[
		(["smact"], {}),
		(["structure"], {}),
		(
			["structure"],
			{"structure": {"threshold_distance": 0.5, "threshold_volume": 0.1}},
		),
		(
			["smact", "structure"],
			{"structure": {"threshold_distance": 0.6, "threshold_volume": 0.2}},
		),
	],
)
def test_Validator(prepare_gen_xtals: list[Crystal], methods: list[str], kwargs: dict):
	"""Test Validator."""
	gen_xtals = prepare_gen_xtals

	validator = Validator(methods=methods, **kwargs)
	dict_individual_scores_1, times_1 = validator.validate(xtals=gen_xtals, skip=[])
	dict_individual_scores_2, times_2 = validator.validate(xtals=gen_xtals, skip=[])
	dict_individual_scores_3, times_3 = validator.validate(
		xtals=gen_xtals, skip=[methods[0]]
	)
	scores_1 = np.ones(len(gen_xtals), dtype=float)
	scores_2 = np.ones(len(gen_xtals), dtype=float)
	scores_3 = np.ones(len(gen_xtals), dtype=float)
	for scores in dict_individual_scores_1.values():
		scores_1 *= scores
	for scores in dict_individual_scores_2.values():
		scores_2 *= scores
	for scores in dict_individual_scores_3.values():
		scores_3 *= scores
	scores_3 *= dict_individual_scores_1[methods[0]]

	assert all(scores_1 == scores_2)
	assert all(scores_1 == scores_3)
	assert len(scores_1) == len(gen_xtals)
	assert all(0.0 <= score <= 1.0 for score in scores_1)
	assert all(isinstance(score, float) for score in scores_1)
	for times in [times_1, times_2, times_3]:
		for item in times.values():
			assert item >= 0.0
