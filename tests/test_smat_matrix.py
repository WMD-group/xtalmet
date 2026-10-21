"""Regression tests for candidate-based StructureMatcher matrices."""

from importlib import import_module
from multiprocessing import get_context
from os import getpid
from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import (
	ElementComparator,
	FrameworkComparator,
	OrderDisorderElementComparator,
	StructureMatcher,
)
from pymatgen.core import Composition, Lattice, Structure

from xtalmet.crystal import Crystal
from xtalmet.distance import distance, distance_matrix
from xtalmet.evaluator import Evaluator

distance_module = import_module("xtalmet.distance")


def _silicon(length=3):
	return Crystal(Lattice.cubic(length), ["Si"], [[0, 0, 0]])


def _exhaustive(sources, targets=None, **settings):
	matcher = StructureMatcher(**settings)
	symmetric = targets is None
	targets = sources if symmetric else targets
	result = np.ones((len(sources), len(targets)))
	for i, source in enumerate(sources):
		for j in range(i + 1 if symmetric else len(targets)):
			result[i, j] = 0.0 if matcher.fit(source, targets[j]) else 1.0
			if symmetric:
				result[j, i] = result[i, j]
	return result


@pytest.fixture
def structures():
	"""Include duplicates, different compositions, polymorphs, and a supercell."""
	silicon = _silicon()
	return [
		silicon,
		_silicon(6),
		Crystal(Lattice.cubic(3), ["Ge"], [[0, 0, 0]]),
		silicon.copy(),
		silicon * [2, 1, 1],
		Crystal(Lattice.cubic(3), ["Si", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
		Crystal.from_file(Path(__file__).parent / "data/wz-ZnO.cif"),
		Crystal.from_file(Path(__file__).parent / "data/rs-ZnO.cif"),
	]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize(
	"settings",
	[
		{},
		{"scale": False},
		{"primitive_cell": False, "attempt_supercell": True},
		{"primitive_cell": False, "allow_subset": True},
		{"ignored_species": ["Li"]},
		{"comparator": ElementComparator()},
		{"comparator": FrameworkComparator()},
	],
)
def test_smat_matches_exhaustive(structures, settings, two_sets, parallel):
	"""Pruning and worker reuse must preserve every distance and the inputs."""
	targets = list(reversed(structures[:6])) if two_sets else None
	before = [structure.as_dict() for structure in structures]
	expected = _exhaustive(structures, targets, **settings)
	result = distance_matrix(
		"smat",
		structures,
		targets,
		multiprocessing=parallel,
		n_processes=2,
		args_dist=settings,
		verbose=True,
	)
	assert isinstance(result[0], np.ndarray)
	assert result[0].dtype == np.float64
	np.testing.assert_array_equal(result[0], expected)
	assert len(result) == (4 if two_sets else 3)
	assert result[-1].keys() == (
		{"emb_1", "emb_2", "d_mtx"} if two_sets else {"emb_1", "d_mtx"}
	)
	assert before == [structure.as_dict() for structure in structures]
	assert [Crystal.from_Structure(s).as_dict() for s in structures] == [
		s.as_dict() for s in result[1]
	]
	if two_sets:
		assert [Crystal.from_Structure(s).as_dict() for s in targets] == [
			s.as_dict() for s in result[2]
		]


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_settings_do_not_leak(parallel):
	"""Each call must honor its own settings, including serial non-defaults."""
	for scale, expected in [(False, 1.0), (True, 0.0), (False, 1.0)]:
		sources, targets = [_silicon()], [_silicon(6)]
		settings = {"scale": scale}
		assert distance("smat", sources[0], targets[0], args_dist=settings) == expected
		actual = distance_matrix(
			"smat",
			sources,
			targets,
			multiprocessing=parallel,
			n_processes=2,
			args_dist=settings,
		)
		np.testing.assert_array_equal(actual, [[expected]])


@pytest.mark.parametrize("two_sets, expected_calls", [(False, 4), (True, 5)])
def test_smat_prunes_calls_and_reuses_matcher(monkeypatch, two_sets, expected_calls):
	"""Different compositions never reach fit; all candidate pairs do."""
	structures = [
		_silicon(),
		_silicon(),
		Crystal(Lattice.cubic(3), ["Ge"], [[0, 0, 0]]),
	]
	matchers = []
	fit = StructureMatcher.fit

	def counted_fit(self, source, target, **kwargs):
		assert (
			source.composition.fractional_composition
			== target.composition.fractional_composition
		)
		matchers.append(self)
		assert kwargs["skip_structure_reduction"] is True
		return fit(self, source, target, **kwargs)

	monkeypatch.setattr(StructureMatcher, "fit", counted_fit)
	actual = distance_matrix("smat", structures, structures if two_sets else None)
	np.testing.assert_array_equal(actual, [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
	assert len(matchers) == expected_calls
	assert all(matcher is matchers[0] for matcher in matchers)


def test_smat_no_candidates_needs_no_pool(monkeypatch):
	"""Disjoint compositions should return a dense mismatch matrix immediately."""

	def unexpected(*args, **kwargs):
		raise AssertionError("No candidate requires a pool or fit")

	monkeypatch.setattr(StructureMatcher, "fit", unexpected)
	monkeypatch.setattr(distance_module, "Pool", unexpected)
	monkeypatch.setattr(distance_module, "_smat_prepare", unexpected)
	actual = distance_matrix(
		"smat",
		[_silicon()],
		[Crystal(Lattice.cubic(3), ["Ge"], [[0, 0, 0]])],
		multiprocessing=True,
		n_processes=2,
	)
	np.testing.assert_array_equal(actual, [[1.0]])


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_nontransitive_matches(parallel):
	"""A match through an intermediate structure must not imply a direct match."""
	structures = [_silicon(length) for length in (3, 3.5, 4)]
	actual = distance_matrix(
		"smat",
		structures,
		multiprocessing=parallel,
		n_processes=2,
		args_dist={"scale": False},
	)
	np.testing.assert_array_equal(actual, [[0, 0, 1], [0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("two_sets", [False, True])
def test_smat_composition_tolerance(two_sets):
	"""Approximate composition equality also requires pairwise comparisons."""
	structures = [
		Crystal(Lattice.cubic(3), [{"Si": 0.5 + delta, "Ge": 0.5 - delta}], [[0, 0, 0]])
		for delta in (
			0,
			0.75 * Composition.amount_tolerance,
			1.5 * Composition.amount_tolerance,
		)
	]
	keys = [structure.composition.fractional_composition for structure in structures]
	assert keys[0] == keys[1] and keys[1] == keys[2] and keys[0] != keys[2]
	targets = structures if two_sets else None
	settings = {
		"primitive_cell": False,
		"comparator": OrderDisorderElementComparator(),
	}
	expected = _exhaustive(structures, targets, **settings)
	assert expected[1, 2] == 0
	# The public Crystal conversion currently rejects disordered structures.
	np.testing.assert_array_equal(
		distance_module._distance_matrix_d_smat(structures, targets, **settings),
		expected,
	)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("two_sets", [False, True])
def test_smat_equal_compositions_with_different_hashes(parallel, two_sets):
	"""Trace species at the equality tolerance must not cause false negatives."""
	tolerance = Composition.amount_tolerance
	structures = [
		Crystal(
			Lattice.cubic(3), [{"Si": 1 - tolerance, trace: tolerance}], [[0, 0, 0]]
		)
		for trace in ("Ge", "C")
	]
	keys = [structure.composition.fractional_composition for structure in structures]
	assert keys[0] == keys[1]
	targets = structures if two_sets else None
	expected = _exhaustive(structures, targets)
	np.testing.assert_array_equal(expected, np.zeros((2, 2)))
	np.testing.assert_array_equal(
		distance_module._distance_matrix_d_smat(
			structures, targets, multiprocessing=parallel, n_processes=2
		),
		expected,
	)


def test_smat_fit_errors_propagate(monkeypatch):
	"""A failed comparison must not become a valid mismatch."""

	def failing_fit(*args, **kwargs):
		raise ValueError("fit failed")

	monkeypatch.setattr(StructureMatcher, "fit", failing_fit)
	with pytest.raises(ValueError, match="fit failed"):
		distance_matrix("smat", [_silicon()])


def test_smat_comparison_direction(monkeypatch):
	"""Square matrices mirror the lower triangle; rectangular fits stay directed."""
	monkeypatch.setattr(
		StructureMatcher,
		"fit",
		lambda self, first, second, **kwargs: first.volume >= second.volume,
	)
	structures = [_silicon(), _silicon(6)]
	np.testing.assert_array_equal(distance_matrix("smat", structures), np.zeros((2, 2)))
	np.testing.assert_array_equal(
		distance_matrix("smat", structures, structures), [[0, 1], [0, 0]]
	)


def test_smat_spawn_workers(monkeypatch):
	"""Worker state must be initialized without relying on fork inheritance."""
	monkeypatch.setattr(distance_module, "Pool", get_context("spawn").Pool)
	actual = distance_matrix(
		"smat",
		[_silicon(), _silicon(6)],
		multiprocessing=True,
		n_processes=2,
		args_dist={"scale": False},
	)
	np.testing.assert_array_equal(actual, [[0, 1], [1, 0]])


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_evaluator_without_cache(parallel):
	"""U/N must use configured matching even without a disk cache."""
	structures = [_silicon(), _silicon(6), _silicon(6)]
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance="smat",
		ref_xtals=[_silicon()],
		args_dist={"scale": False},
	)
	overall, individual, _ = evaluator.evaluate(
		structures,
		multiprocessing=parallel,
		n_processes=2,
	)
	np.testing.assert_array_equal(individual, [0, 1, 0])
	assert overall == pytest.approx(1 / 3)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("two_sets", [False, True])
def test_smat_reduces_participating_structures_once(monkeypatch, parallel, two_sets):
	"""Reduction happens in the parent once per participating input position."""
	sources = [_silicon(), _silicon(6), Crystal(Lattice.cubic(3), ["Ge"], [[0, 0, 0]])]
	targets = (
		[_silicon(), _silicon(5), Crystal(Lattice.cubic(3), ["Ne"], [[0, 0, 0]])]
		if two_sets
		else None
	)
	expected = _exhaustive(sources, targets)
	parent_pid = getpid()
	reduced = []
	reduce = Structure.get_reduced_structure

	def counted_reduce(self, *args, **kwargs):
		assert getpid() == parent_pid, "Workers must reuse the parent's reductions"
		reduced.append(self.composition.reduced_formula)
		return reduce(self, *args, **kwargs)

	monkeypatch.setattr(Structure, "get_reduced_structure", counted_reduce)
	actual = distance_matrix(
		"smat", sources, targets, multiprocessing=parallel, n_processes=2
	)
	np.testing.assert_array_equal(actual, expected)
	assert reduced == (["Si"] * 4 if two_sets else ["Si", "Si", "Ge"])


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_prunes_unequal_reduced_site_counts(monkeypatch, parallel):
	"""Without subset/supercell matching, unequal reduced site counts cannot fit."""

	def unexpected(*args, **kwargs):
		raise AssertionError("Unequal site counts should not reach fit")

	monkeypatch.setattr(StructureMatcher, "fit", unexpected)
	actual = distance_matrix(
		"smat",
		[_silicon()],
		[_silicon() * [2, 1, 1]],
		args_dist={"primitive_cell": False},
		multiprocessing=parallel,
		n_processes=2,
	)
	np.testing.assert_array_equal(actual, [[1.0]])


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_removes_ignored_species_before_reduction(parallel):
	"""Removing Li exposes a smaller primitive Si cell before site-count pruning."""
	source = Crystal(
		Lattice.cubic(3),
		["Si", "Si", "Li"],
		[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
	)
	target = source.copy()
	target.remove_species(["Li"])
	settings = {"ignored_species": ["Li"], "scale": False}
	expected = _exhaustive([source], [target], **settings)
	np.testing.assert_array_equal(expected, [[0.0]])
	actual = distance_matrix(
		"smat",
		[source],
		[target],
		args_dist=settings,
		multiprocessing=parallel,
		n_processes=2,
	)
	np.testing.assert_array_equal(actual, expected)
	assert len(source) == 3 and "Li" in source.composition


def test_smat_niggli_reduction_without_primitive_cell(monkeypatch):
	"""Disabling primitive cells must still normalize a skewed lattice basis."""
	source = _silicon()
	source.make_supercell([[1, 3, 0], [0, 1, 0], [0, 0, 1]])
	fit = StructureMatcher.fit

	def checked_fit(self, first, second, **kwargs):
		np.testing.assert_allclose(first.lattice.abc, [3, 3, 3])
		np.testing.assert_allclose(second.lattice.abc, [3, 3, 3])
		return fit(self, first, second, **kwargs)

	def unexpected(*args, **kwargs):
		raise AssertionError("Primitive reduction was disabled")

	monkeypatch.setattr(StructureMatcher, "fit", checked_fit)
	monkeypatch.setattr(Structure, "get_primitive_structure", unexpected)
	actual = distance_matrix(
		"smat", [source], [_silicon()], args_dist={"primitive_cell": False}
	)
	np.testing.assert_array_equal(actual, [[0.0]])


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("two_sets", [False, True])
def test_smat_falls_back_when_reduction_changes_composition(parallel, two_sets):
	"""A second composition check after reduction must not reject an original match."""
	delta = 2 * Composition.amount_tolerance
	source = Structure(
		Lattice.cubic(3),
		[
			{"Si": 0.5 + delta, "Ge": 0.5 - delta},
			{"Si": 0.5 - delta, "Ge": 0.5 + delta},
		],
		[[0, 0, 0], [0.5, 0.5, 0.5]],
	)
	target = Structure(
		Lattice.cubic(3), [{"Si": 0.5, "Ge": 0.5}] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]]
	)
	settings = {"comparator": OrderDisorderElementComparator()}
	matcher = StructureMatcher(**settings)
	assert matcher.fit(source, target)
	assert not matcher.fit(
		source.get_reduced_structure().get_primitive_structure(),
		target.get_reduced_structure().get_primitive_structure(),
		skip_structure_reduction=True,
	)
	sources = [source, target]
	targets = list(reversed(sources)) if two_sets else None
	expected = _exhaustive(sources, targets, **settings)
	# The public conversion still rejects partially occupied structures.
	actual = distance_module._distance_matrix_d_smat(
		sources, targets, multiprocessing=parallel, n_processes=2, **settings
	)
	np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("parallel", [False, True])
def test_smat_reduction_errors_propagate(monkeypatch, parallel):
	"""Failed preparation must abort before starting a pool or producing distances."""

	def failing_reduction(*args, **kwargs):
		raise ValueError("reduction failed")

	def unexpected(*args, **kwargs):
		raise AssertionError("Preparation failed before a pool was needed")

	monkeypatch.setattr(Structure, "get_reduced_structure", failing_reduction)
	monkeypatch.setattr(distance_module, "Pool", unexpected)
	with pytest.raises(ValueError, match="reduction failed"):
		distance_matrix("smat", [_silicon()], multiprocessing=parallel, n_processes=2)
