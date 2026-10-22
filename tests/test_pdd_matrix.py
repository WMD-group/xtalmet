"""Check bounded PDD blocks against AMD's complete matrix implementation."""

import importlib
import weakref
from contextlib import contextmanager

import amd
import numpy as np
import pytest
from joblib import parallel_config
from joblib._parallel_backends import ThreadingBackend
from joblib.parallel import get_active_backend
from scipy.spatial.distance import squareform

from xtalmet.distance import distance_matrix

distance_module = importlib.import_module("xtalmet.distance")


def pdds(n):
	"""Small distributions with unequal weights and two distance coordinates."""
	return [
		np.array([[0.25, 1 + i / 10, 2.0], [0.75, 2.0, 4 + i / 10]]) for i in range(n)
	]


@pytest.mark.parametrize("backend", ["multiprocessing", "threading", "loky", None])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("args", [{}, {"metric": "minkowski", "p": 1.5}])
def test_pdd_backend_matrix(backend, two_sets, args):
	"""Honor the requested backend and metric without changing any distance."""
	left = pdds(4)
	right = pdds(3)[::-1] if two_sets else None
	expected = (
		amd.PDD_cdist(left, right, n_jobs=1, **args)
		if two_sets
		else squareform(amd.PDD_pdist(left, n_jobs=1, **args))
	)
	actual = distance_matrix(
		"pdd", left, right, args_dist={**args, "backend": backend, "n_jobs": 2}
	)
	np.testing.assert_allclose(
		actual, expected / (1 + expected), rtol=1e-13, atol=1e-13
	)


@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("n_jobs", [None, 0, 1, -1, -2, 2])
def test_pdd_bounded_dispatch(monkeypatch, two_sets, n_jobs):
	"""Keep legacy worker selection, pair direction, and a finite result window."""
	contexts, batches, calls, previous = [], [], [], []

	@contextmanager
	def config(**kwargs):
		contexts.append(kwargs)
		yield

	class Parallel:
		def __enter__(self):
			return self

		def __exit__(self, *args):
			pass

		def __call__(self, tasks):
			assert all(ref() is None for ref in previous)
			jobs = list(tasks)
			assert 0 < len(jobs) <= 4
			batches.append(len(jobs))
			results = [function(*args, **kwargs) for function, args, kwargs in jobs]
			previous[:] = [weakref.ref(result) for result in results]
			return results

	def cdist(left, right, **kwargs):
		assert kwargs == {"n_jobs": 1}
		assert len(left) <= 128 and len(right) <= 128
		calls.append((len(left), len(right)))
		return np.array([[10 * a[0, 1] + b[0, 1] for b in right] for a in left])

	def pdist(left, **kwargs):
		assert kwargs == {"n_jobs": 1}
		assert len(left) <= 128
		calls.append((len(left), len(left)))
		return np.array(
			[10 * a[0, 1] + b[0, 1] for i, a in enumerate(left) for b in left[i + 1 :]]
		)

	monkeypatch.setattr(distance_module, "parallel_config", config)
	monkeypatch.setattr(distance_module, "Parallel", Parallel)
	monkeypatch.setattr(distance_module, "effective_n_jobs", lambda: 2)
	monkeypatch.setattr(amd, "PDD_cdist", cdist)
	monkeypatch.setattr(amd, "PDD_pdist", pdist)
	left = [np.array([[1.0, float(i)]]) for i in range(259)]
	right = left[:133][::-1] if two_sets else None
	actual = distance_matrix(
		"pdd",
		left,
		right,
		normalize=False,
		args_dist={"n_jobs": n_jobs, "backend": "threading"},
	)
	parallel = n_jobs is not None and (n_jobs not in (0, 1) if two_sets else n_jobs > 1)
	assert contexts == [
		{
			"backend": "threading" if parallel else "sequential",
			"n_jobs": n_jobs if parallel else 1,
		}
	]
	assert len(batches) > 1 and len(calls) == sum(batches)
	expected = (
		[[10 * a[0, 1] + b[0, 1] for b in right] for a in left]
		if two_sets
		else [
			[0 if i == j else 10 * min(i, j) + max(i, j) for j in range(259)]
			for i in range(259)
		]
	)
	np.testing.assert_array_equal(actual, expected)


def test_pdd_explicit_jobs_override(monkeypatch):
	"""An explicit serial n_jobs must override the public parallelism switch."""
	monkeypatch.setattr(
		distance_module,
		"_set_n_processes",
		lambda *a: pytest.fail("explicit n_jobs ignored"),
	)
	distance_matrix(
		"pdd", pdds(2), multiprocessing=True, n_processes=2, args_dist={"n_jobs": 1}
	)


def test_pdd_inherits_active_backend(monkeypatch):
	"""backend=None must keep an enclosing joblib backend, as AMD did."""
	parallel = distance_module.Parallel
	selected = []

	def inherited_parallel():
		backend, n_jobs = get_active_backend()
		assert isinstance(backend, ThreadingBackend)
		assert n_jobs == 2
		selected.append(backend)
		return parallel()

	monkeypatch.setattr(distance_module, "Parallel", inherited_parallel)
	left, right = pdds(3), pdds(2)
	expected = amd.PDD_cdist(left, right, n_jobs=1)
	with parallel_config(backend="threading", n_jobs=2):
		actual = distance_matrix(
			"pdd",
			left,
			right,
			normalize=False,
			args_dist={"backend": None, "n_jobs": 2},
		)
	assert len(selected) == 1
	np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("parallel", [False, True])
def test_pdd_block_boundary(parallel):
	"""Check a real backend across a row-block boundary."""
	left, right = pdds(130), pdds(3)
	expected = amd.PDD_cdist(left, right, n_jobs=1)
	actual = distance_matrix(
		"pdd", left, right, normalize=False, multiprocessing=parallel, n_processes=2
	)
	np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("backend", ["multiprocessing", "threading"])
def test_pdd_worker_error(backend):
	"""Errors from inside a block must reach the caller."""
	with pytest.raises(ValueError, match="Unknown Distance Metric"):
		distance_matrix(
			"pdd",
			pdds(2),
			args_dist={"backend": backend, "n_jobs": 2, "metric": "invalid"},
		)


def test_pdd_verbose(capsys):
	"""Report completed pairs once in the parent, ignoring transport requests."""
	distance_matrix(
		"pdd", pdds(3), args_dist={"verbose": True, "return_transport": True}
	)
	assert "Compared 3 PDD pairs" in capsys.readouterr().out
