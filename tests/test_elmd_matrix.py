"""Check parsed ElMD reuse, comparison direction, and bounded worker tasks."""

import importlib
import weakref
from multiprocessing import get_context

import numpy as np
import pytest
from ElMD import ElMD

from xtalmet.distance import _d_elmd, distance_matrix

distance_module = importlib.import_module("xtalmet.distance")


@pytest.mark.parametrize("metric", ["mod_petti", "fast", "mendeleev"])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_elmd_matrix(metric, two_sets, parallel):
	"""Parsing once must retain scalar distances and the original embeddings."""
	left = ["NaCl", "SiO2", "NaCl", "GaN"]
	right = ["ZnO", "NaCl", "Si"] if two_sets else None
	target = left if right is None else right
	expected = np.array([[_d_elmd(a, b, metric=metric) for b in target] for a in left])
	result = distance_matrix(
		"elmd",
		left,
		right,
		multiprocessing=parallel,
		n_processes=2,
		args_dist={"metric": metric},
		verbose=True,
	)
	np.testing.assert_allclose(
		result[0], expected / (1 + expected), rtol=1e-13, atol=1e-13
	)
	assert result[1] == left
	if two_sets:
		assert result[2] == right


def test_elmd_parses_each_formula_once(monkeypatch):
	"""Share parsed strings across both sets, retaining supplied ElMD objects."""
	parsed = []

	def parse(formula, **kwargs):
		parsed.append((formula, kwargs))
		return ElMD(formula, **kwargs)

	monkeypatch.setattr(distance_module, "ElMD", parse)
	obj = ElMD("O", metric="mendeleev")
	left, right = ["Si", "Si", obj], ["Si", "NaCl", obj]
	expected = np.array([[_d_elmd(a, b, metric="fast") for b in right] for a in left])
	actual = distance_matrix(
		"elmd", left, right, normalize=False, args_dist={"metric": "fast"}
	)
	np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
	assert parsed == [("Si", {"metric": "fast"}), ("NaCl", {"metric": "fast"})]
	assert obj.metric == "mendeleev"


@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_elmd_direction_and_bounded_tasks(monkeypatch, two_sets, parallel):
	"""Check finite dispatch and asymmetric values, including each diagonal."""
	monkeypatch.setattr(distance_module, "ElMD", lambda value, **kwargs: int(value))
	monkeypatch.setattr(distance_module, "_d_elmd", lambda a, b, **kwargs: 10 * a + b)
	batches, previous = [], []

	class Pool:
		def __init__(self, processes, initializer, initargs):
			assert processes == 2
			initializer(*initargs)

		def __enter__(self):
			return self

		def __exit__(self, *args):
			pass

		def imap_unordered(self, worker, tasks):
			assert all(ref() is None for ref in previous)
			assert isinstance(tasks, tuple) and 0 < len(tasks) <= 4
			assert all(0 < stop - start <= 1024 for _, start, stop in tasks)
			batches.append(tasks)
			results = [worker(task) for task in reversed(tasks)]
			previous[:] = [weakref.ref(values) for _, _, values in results]
			return iter(results)

	monkeypatch.setattr(distance_module, "Pool", Pool)
	left = [str(i) for i in range(11)]
	right = [str(i % 7) for i in range(2053)] if two_sets else None
	actual = distance_matrix(
		"elmd", left, right, normalize=False, multiprocessing=parallel, n_processes=2
	)
	if two_sets:
		expected = [[10 * int(a) + int(b) for b in right] for a in left]
	else:
		expected = [[10 * max(i, j) + min(i, j) for j in range(11)] for i in range(11)]
	np.testing.assert_array_equal(actual, expected)
	assert len(batches) > 1 if parallel else not batches


def test_elmd_spawn(monkeypatch):
	"""Initialize parsed objects correctly without fork inheritance."""
	monkeypatch.setattr(distance_module, "Pool", get_context("spawn").Pool)
	left, right = ["ZnO", "GaN", "ZnO"], ["Si", "NaCl"]
	expected = distance_matrix("elmd", left, right)
	actual = distance_matrix("elmd", left, right, multiprocessing=True, n_processes=2)
	np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("parallel", [False, True])
def test_elmd_worker_error(parallel):
	"""Propagate backend argument errors in serial and parallel execution."""
	with pytest.raises(TypeError, match="unexpected"):
		distance_matrix(
			"elmd",
			["Si", "O"],
			multiprocessing=parallel,
			n_processes=2,
			args_dist={"unsupported_argument": True},
		)
