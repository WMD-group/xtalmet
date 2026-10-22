"""Check grouped exact distances against the original pairwise definitions."""

import importlib

import numpy as np
import pytest

from xtalmet.crystal import Crystal
from xtalmet.distance import _EmbeddingError, distance, distance_matrix
from xtalmet.evaluator import Evaluator

distance_module = importlib.import_module("xtalmet.distance")


@pytest.mark.parametrize("metric", ["comp", "wyckoff"])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_matrix(metric, two_sets, parallel, reverse, monkeypatch):
	"""Preserve every match, input position, failure, and verbose embedding."""
	a, b = (
		((("Si", 1),), (("O", 1),))
		if metric == "comp"
		else ((1, ("a", "a")), (2, ("b",)))
	)
	error = _EmbeddingError(RuntimeError("failed"))
	left = [error, a, b, a, error, b]
	right = [b, error, a, a] if two_sets else None
	if reverse:
		left.reverse()
	target = left if right is None else right
	expected = np.array([[distance(metric, x, y) for y in target] for x in left])
	monkeypatch.setattr(
		distance_module, "Pool", lambda **k: pytest.fail("matrix started a pool")
	)
	result = distance_matrix(
		metric, left, right, multiprocessing=parallel, n_processes=2, verbose=True
	)
	np.testing.assert_array_equal(result[0], expected)
	assert result[0].dtype == np.float64
	assert result[1] == left
	if two_sets:
		assert result[2] == right
	assert result[-1]["emb_1"] == 0


@pytest.mark.parametrize("metric", ["comp", "wyckoff"])
@pytest.mark.parametrize("repeated", [False, True])
def test_exact_unique_or_repeated(metric, repeated):
	"""Handle both extreme group sizes without omitting matching positions."""
	values = [
		(("Si", i + 1),) if metric == "comp" else (i + 1, ("a",)) for i in range(9)
	]
	if repeated:
		values = [values[0]] * len(values)
	expected = np.zeros((9, 9)) if repeated else 1 - np.eye(9)
	np.testing.assert_array_equal(distance_matrix(metric, values), expected)


@pytest.mark.parametrize("metric", ["comp", "wyckoff"])
@pytest.mark.parametrize("cache", [False, True])
def test_exact_evaluator_keeps_dense_matrices(metric, cache, tmp_path, monkeypatch):
	"""Scores must still use complete dense matrices, even without disk caching."""
	xtals = [
		Crystal(np.eye(3) * 4, ["Si"], [[0, 0, 0]]),
		Crystal(np.eye(3) * 4, ["O"], [[0, 0, 0]]),
		Crystal(np.eye(3) * 4, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
	]
	xtals.append(xtals[0])
	matrices = []
	evaluator_module = importlib.import_module("xtalmet.evaluator")

	def capture(*args, **kwargs):
		result = distance_matrix(*args, **kwargs)
		matrices.append(result[0])
		return result

	monkeypatch.setattr(evaluator_module, "distance_matrix", capture)
	evaluator = Evaluator(
		uniqueness=True, novelty=True, distance=metric, ref_xtals=[xtals[0]]
	)
	directory = str(tmp_path) if cache else None
	_, scores, _ = evaluator.evaluate(xtals, dir_intermediate=directory)
	assert [m.shape for m in matrices] == [(4, 4), (4, 1)]
	assert all(m.dtype == np.float64 for m in matrices)
	uni, nov = matrices
	expected = [np.all(uni[i, :i] != 0) * np.all(nov[i] != 0) for i in range(4)]
	np.testing.assert_array_equal(scores, expected)
	if cache:
		assert len(list(tmp_path.rglob(f"mtx_*_{metric}.pkl.gz"))) == 2
		_, repeated, times = evaluator.evaluate(xtals, dir_intermediate=directory)
		np.testing.assert_array_equal(repeated, scores)
		assert len(matrices) == 2
		assert times["uni_d_mtx"] == times["nov_d_mtx"] == 0
