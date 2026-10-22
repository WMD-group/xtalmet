"""Test validators."""

import gzip
import pickle

import numpy as np
import pytest
from pymatgen.core import Composition

from xtalmet.crystal import Crystal
from xtalmet.evaluator import Evaluator
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
	"formula, expected",
	[("TiO", 1.0), ("NaCl", 1.0), ("NaCl2", 0.0), ("Fe3O4", 0.0)],
)
def test_smact_4_defaults(formula, expected):
	"""Use low commonality and leave mixed valence disabled, including in V."""
	species = [
		str(element)
		for element, count in Composition(formula).items()
		for _ in range(int(count))
	]
	coordinates = np.arange(len(species))[:, None] * np.ones((1, 3)) / len(species)
	xtal = Crystal(np.eye(3) * 10, species, coordinates)

	np.testing.assert_array_equal(SMACTValidator().validate([xtal]), [expected])
	overall, scores, _ = Evaluator(validity=["smact"]).evaluate(
		[xtal], dir_intermediate=None
	)
	np.testing.assert_array_equal(scores, [expected])
	assert overall == expected


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


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("threshold", [0.5, 0.6])
@pytest.mark.parametrize("delta", [-1e-9, 0.0, 1e-9])
def test_structure_periodic_self_distance(skew, threshold, delta):
	"""Include self images, even when the shortest translation combines axes."""
	distance = threshold + delta
	lattice = (
		[[2, 0, 0], [2, distance, 0], [0, 0, 2]] if skew else np.diag([distance, 2, 2])
	)
	xtal = Crystal(lattice, ["Si"], [[0, 0, 0]])
	scores = StructureValidator(threshold_distance=threshold).validate([xtal])
	np.testing.assert_array_equal(scores, [float(delta >= 0)])


@pytest.mark.parametrize("across_boundary", [False, True])
@pytest.mark.parametrize("distance", [0.0, 0.5 - 1e-9, 0.5, 0.5 + 1e-9, 1.0])
def test_structure_distinct_sites(distance, across_boundary):
	"""Reject close or overlapping distinct sites, including across cell edges."""
	x = 4 - distance if across_boundary else distance
	xtal = Crystal(
		np.eye(3) * 4,
		["Si", "Si"],
		[[0, 0, 0], [x, 0, 0]],
		coords_are_cartesian=True,
	)
	scores = StructureValidator().validate([xtal])
	np.testing.assert_array_equal(scores, [float(distance >= 0.5)])


@pytest.mark.parametrize("length, expected", [(0.25, 0.0), (0.5, 1.0), (1.0, 1.0)])
def test_structure_volume_boundary(length, expected):
	"""Keep the volume cutoff inclusive while checking periodic distances."""
	xtal = Crystal(np.eye(3) * length, ["Si"], [[0, 0, 0]])
	validator = StructureValidator(threshold_distance=0.1, threshold_volume=0.125)
	np.testing.assert_array_equal(validator.validate([xtal]), [expected])


@pytest.mark.parametrize("threshold", [0.0, -0.5])
def test_structure_nonpositive_distance_threshold(threshold):
	"""Nonpositive distance thresholds still allow overlap and enforce volume."""
	xtals = [
		Crystal(np.eye(3), ["Si", "Si"], [[0, 0, 0], [0, 0, 0]]),
		Crystal(np.eye(3) * 0.25, ["Si"], [[0, 0, 0]]),
	]
	validator = StructureValidator(threshold_distance=threshold)
	np.testing.assert_array_equal(validator.validate(xtals), [1.0, 0.0])


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_structure_nonfinite_coordinates(coordinate, threshold):
	"""Invalid generated coordinates must not abort or pass a validation batch."""
	valid = Crystal(np.eye(3) * 3, ["Si"], [[0, 0, 0]])
	invalid = Crystal(np.eye(3) * 3, ["Si", "Si"], [[0, 0, 0], [coordinate, 0, 0]])
	validator = StructureValidator(threshold_distance=threshold)
	np.testing.assert_array_equal(
		validator.validate([valid, invalid, valid]), [1.0, 0.0, 1.0]
	)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_structure_nonfinite_distance_threshold(threshold):
	"""Nonfinite cutoffs must not enter native neighbor search or pass screening."""
	xtal = Crystal(np.eye(3) * 3, ["Si"], [[0, 0, 0]])
	validator = StructureValidator(threshold_distance=threshold)
	np.testing.assert_array_equal(validator.validate([xtal]), [0.0])


@pytest.mark.parametrize("threshold, expected", [(0.5, [1, 0, 1]), (0.6, [0, 0, 1])])
@pytest.mark.parametrize("agg_func", ["prod", "ave"])
def test_structure_validity_integration(threshold, expected, agg_func):
	"""Propagate periodic validity and custom settings through both public callers."""
	xtals = [
		Crystal(np.diag([0.5, 2, 2]), ["Si"], [[0, 0, 0]]),
		Crystal([[2, 0, 0], [2, 0.4, 0], [0, 0, 2]], ["Si"], [[0, 0, 0]]),
		Crystal(np.eye(3) * 3, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
	]
	settings = {"structure": {"threshold_distance": threshold}}
	validator = Validator(methods=["structure"], **settings)
	by_method, _ = validator.validate(xtals, skip=[])
	np.testing.assert_array_equal(by_method["structure"], expected)
	assert by_method["structure"].dtype == np.dtype(float)

	evaluator = Evaluator(
		validity=["structure"], agg_func=agg_func, args_validity=settings
	)
	overall, scores, _ = evaluator.evaluate(xtals, dir_intermediate=None)
	np.testing.assert_array_equal(scores, expected)
	assert scores.dtype == np.dtype(float)
	assert overall == pytest.approx(np.mean(expected))


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
