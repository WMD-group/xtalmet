"""Check rowwise normalization and blending, including caller-owned buffers."""

import importlib
import weakref

import numpy as np
import pytest

from xtalmet.distance import _EmbeddingError, distance, distance_matrix

distance_module = importlib.import_module("xtalmet.distance")


@pytest.mark.parametrize("metric", ["magpie", "amd", "elmd", "pdd", "elmd+amd"])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_normalized_matrix(metric, two_sets, normalize, monkeypatch):
	"""Keep scalar results and failure positions without full-matrix arithmetic."""
	a, b = {
		"magpie": ([1.0, 2.0], [3.0, 4.0]),
		"amd": ([1.0, 2.0], [3.0, 4.0]),
		"elmd": ("Si", "O"),
		"pdd": (np.array([[1.0, 1.0, 2.0]]), np.array([[1.0, 3.0, 4.0]])),
		"elmd+amd": (("Si", [1.0, 2.0]), ("O", [3.0, 4.0])),
	}[metric]
	error = _EmbeddingError(RuntimeError("failed"))
	left = [a, error, b, a]
	right = [error, b, a] if two_sets else None
	target = left if right is None else right
	args = {"coefs": {"elmd": 2, "amd": 3}} if metric == "elmd+amd" else {}
	expected = np.array(
		[
			[
				distance(metric, x, y, normalize=normalize, args_dist=args)
				for y in target
			]
			for x in left
		]
	)
	original = distance_module._normalize

	def normalize_row(value):
		assert np.ndim(value) <= 1
		return original(value)

	monkeypatch.setattr(distance_module, "_normalize", normalize_row)
	actual = distance_matrix(metric, left, right, normalize=normalize, args_dist=args)
	np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True)


@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize(
	"coefs", [{"elmd": 0, "amd": 1}, {"elmd": 3, "amd": 0}, {"elmd": 2, "amd": 3}]
)
def test_blend_reuses_elmd_only(monkeypatch, two_sets, coefs):
	"""Reuse the owned component without mutating raw AMD output or its alias."""
	left = [("Si", [1.0, 2.0]), ("O", [3.0, 4.0]), ("Si", [2.0, 3.0])]
	right = left[:2][::-1] if two_sets else None
	target = left if right is None else right
	out = np.empty((3, 2) if two_sets else (3,))
	args = {"coefs": coefs, "amd": {"out": out}}
	expected = np.array(
		[
			[distance("elmd+amd", a, b, args_dist={"coefs": coefs}) for b in target]
			for a in left
		]
	)
	raw = np.array(
		[[distance("amd", a[1], b[1], normalize=False) for b in target] for a in left]
	)
	original = distance_module._distance_matrix_d_elmd
	owned = []

	def capture(*args, **kwargs):
		matrix = original(*args, **kwargs)
		owned.append(matrix)
		return matrix

	monkeypatch.setattr(distance_module, "_distance_matrix_d_elmd", capture)
	actual = distance_matrix("elmd+amd", left, right, args_dist=args)
	assert actual is owned[0]
	assert not np.shares_memory(actual, out)
	np.testing.assert_array_equal(out, raw if two_sets else raw[np.triu_indices(3, 1)])
	np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("two_sets", [False, True])
def test_blend_releases_amd_before_restoration(monkeypatch, two_sets):
	"""A row view must not retain the whole AMD component during restoration."""
	a, b = ("Si", [1.0, 2.0]), ("O", [3.0, 4.0])
	left = [a, _EmbeddingError(RuntimeError("failed")), b]
	right = [b, a] if two_sets else None
	shape = (3, 2) if two_sets else (3, 3)
	original_amd = distance_module._distance_matrix_d_amd
	original_full = np.full
	components = []

	def capture(*args, **kwargs):
		matrix = original_amd(*args, **kwargs)
		components.append(weakref.ref(matrix))
		return matrix

	def restore(array_shape, value, **kwargs):
		if array_shape == shape:
			assert components and components[0]() is None
		return original_full(array_shape, value, **kwargs)

	monkeypatch.setattr(distance_module, "_distance_matrix_d_amd", capture)
	monkeypatch.setattr(np, "full", restore)
	distance_matrix("elmd+amd", left, right)
