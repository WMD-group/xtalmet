"""Check Magpie matrices against the single-pair distance and VSUN formulas."""

import numpy as np
import pytest

from xtalmet.crystal import Crystal
from xtalmet.distance import (
	_d_amd,
	_d_magpie,
	_distance_matrix_d_magpie,
	distance_matrix,
)
from xtalmet.evaluator import Evaluator


@pytest.fixture
def embeddings():
	"""Provide distinct, duplicate, and cross-set matching feature vectors."""
	rng = np.random.default_rng(7)
	left = rng.normal(size=(5, 132)).tolist()
	right = rng.normal(size=(3, 132)).tolist()
	left[2] = left[0].copy()
	right[1] = left[4].copy()
	return left, right


@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_magpie_matrix(embeddings, two_sets, parallel, normalize):
	"""Preserve complete matrices, normalization, order, and verbose embeddings."""
	left, right = embeddings
	target = right if two_sets else left
	expected = np.array([[_d_magpie(a, b) for b in target] for a in left])
	if normalize:
		expected /= 1 + expected
	result = distance_matrix(
		"magpie",
		left,
		right if two_sets else None,
		normalize=normalize,
		multiprocessing=parallel,
		n_processes=2,
		verbose=True,
	)
	np.testing.assert_allclose(result[0], expected, rtol=1e-13, atol=1e-13)
	assert result[0].dtype == np.dtype(float)
	assert result[1] == left
	if two_sets:
		assert result[2] == right
	assert result[-1]["d_mtx"] >= 0
	assert result[-1]["emb_1"] == 0
	if not two_sets:
		np.testing.assert_array_equal(np.diag(result[0]), np.zeros(len(left)))
		np.testing.assert_array_equal(result[0], result[0].T)


@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_magpie_nonfinite_features(two_sets, parallel, normalize):
	"""Keep NaN/inf pair results, including nonfinite self distances."""
	left = [[0.0, 1.0], [np.nan, 2.0], [np.inf, 1.0]]
	right = [[1.0, 0.0], [np.nan, 0.0]] if two_sets else None
	target = right if two_sets else left
	with np.errstate(invalid="ignore"):
		expected = np.array([[_d_magpie(a, b) for b in target] for a in left])
		if normalize:
			expected /= 1 + expected
		actual = distance_matrix(
			"magpie",
			left,
			right,
			normalize=normalize,
			multiprocessing=parallel,
			n_processes=2,
		)
	np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("parallel", [False, True])
def test_magpie_singleton(parallel):
	"""A single embedding still produces a dense 1-by-1 matrix."""
	actual = _distance_matrix_d_magpie(
		[[1.0, 2.0]], multiprocessing=parallel, n_processes=2
	)
	np.testing.assert_array_equal(actual, [[0.0]])


@pytest.mark.parametrize("parallel", [False, True])
def test_magpie_evaluator(parallel):
	"""Keep U/N aggregation for real crystal embeddings without disk caching."""
	xtals = [
		Crystal(np.eye(3) * 4, species, [[0, 0, 0], [0.5, 0.5, 0.5]])
		for species in (["Zn", "O"], ["Ga", "N"], ["Zn", "O"], ["Si", "Si"])
	]
	embs = [xtal.get_embedding("magpie") for xtal in xtals]
	distances = np.array([[_d_magpie(a, b) for b in embs] for a in embs])
	distances /= 1 + distances
	expected = distances.sum(axis=1) / (len(xtals) - 1) * distances[:, 1]
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance="magpie",
		ref_xtals=[xtals[1]],
		multiprocessing=parallel,
		n_processes=2,
	)
	overall, scores, _ = evaluator.evaluate(
		xtals, dir_intermediate=None, multiprocessing=parallel, n_processes=2
	)
	np.testing.assert_allclose(scores, expected, rtol=1e-13, atol=1e-13)
	assert overall == pytest.approx(expected.mean(), rel=1e-13, abs=1e-13)


def test_amd_output_buffer_stays_unnormalized():
	"""Magpie's memory optimization must not mutate caller-owned AMD output."""
	left = [[1.0, 2.0], [3.0, 4.0]]
	right = [[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]]
	raw = np.array([[_d_amd(a, b) for b in right] for a in left])
	output = np.empty(raw.shape)
	actual = distance_matrix("amd", left, right, args_dist={"out": output})
	np.testing.assert_array_equal(output, raw)
	np.testing.assert_allclose(actual, raw / (1 + raw))
	assert not np.shares_memory(actual, output)
