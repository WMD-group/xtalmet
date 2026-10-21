"""This module offers a range of distance functions for crystals."""

import functools
import time
import warnings
from collections.abc import Callable, Iterator
from itertools import chain
from multiprocessing import Pool, cpu_count
from typing import Literal, overload

import amd
import numpy as np
from ElMD import elmd
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Composition, Structure
from scipy.spatial.distance import cdist, squareform

from .constants import (
	CONTINUOUS_UNNORMALIZED_DISTANCES,
	DIST_WO_EMB,
	ELMD_AMD_COEF_AMD,
	ELMD_AMD_COEF_ELMD,
	TYPE_EMB_ALL,
	TYPE_EMB_AMD,
	TYPE_EMB_COMP,
	TYPE_EMB_ELMD,
	TYPE_EMB_MAGPIE,
	TYPE_EMB_PDD,
	TYPE_EMB_WYCKOFF,
)
from .crystal import Crystal, _to_crystal, _to_crystal_list


class _EmbeddingError:
	"""Sentinel wrapping a failed embedding computation."""

	def __init__(self, exc: Exception) -> None:
		self.exc = exc


def _normalize(d: float) -> float:
	"""Normalize d to [0, 1] via d / (1 + d).

	Args:
		d (float): Distance value.

	Returns:
		float: Normalized distance.
	"""
	return d / (1 + d)


def _resolve_elmd_amd_coefs(args_dist: dict) -> tuple[float, float]:
	"""Read and normalize elmd+amd blending coefficients from args_dist.

	Args:
		args_dist (dict): Distance arguments dict, optionally containing "coefs".

	Returns:
		tuple[float, float]: Normalized (coef_elmd, coef_amd).
	"""
	if "coefs" in args_dist:
		coefs = args_dist["coefs"]
		coef_elmd, coef_amd = coefs["elmd"], coefs["amd"]
		total = coef_elmd + coef_amd
		return coef_elmd / total, coef_amd / total
	return ELMD_AMD_COEF_ELMD, ELMD_AMD_COEF_AMD


def _insert_nan_rows_cols(
	d_mtx: np.ndarray,
	error_indices_1: list,
	error_indices_2: list | None,
) -> np.ndarray:
	"""Insert NaN rows and columns for failed embeddings.

	Args:
		d_mtx (np.ndarray): Distance matrix computed from valid embeddings only.
		error_indices_1 (list): Row indices where embeddings failed.
		error_indices_2 (list | None): Column indices where embeddings failed.
			If None, the same indices as error_indices_1 are used (symmetric case).

	Returns:
		np.ndarray: Distance matrix with NaN rows/columns inserted.
	"""
	for i in error_indices_1:
		d_mtx = np.insert(d_mtx, i, float("nan"), axis=0)
	col_indices = error_indices_2 if error_indices_2 is not None else error_indices_1
	for j in col_indices:
		d_mtx = np.insert(d_mtx, j, float("nan"), axis=1)
	return d_mtx


def _blend_elmd_amd_distance(
	d_elmd_unnorm: float | np.ndarray,
	d_amd_unnorm: float | np.ndarray,
	args_dist: dict,
) -> float | np.ndarray:
	"""Blend normalized elmd and amd distances.

	Args:
		d_elmd_unnorm (float | np.ndarray): Unnormalized ElMD distance(s).
		d_amd_unnorm (float | np.ndarray): Unnormalized AMD distance(s).
		args_dist (dict): Distance arguments, optionally containing "coefs".

	Returns:
		float | np.ndarray: Blended distance(s).
	"""
	coef_elmd, coef_amd = _resolve_elmd_amd_coefs(args_dist)
	return coef_elmd * _normalize(d_elmd_unnorm) + coef_amd * _normalize(d_amd_unnorm)


def _d_smat(
	emb_1: Crystal, emb_2: Crystal, matcher: StructureMatcher | None = None, **kwargs
) -> float:
	"""Compute the binary distance based on pymatgen's StructureMatcher.

	Args:
		emb_1 (Crystal): Embedding 1.
		emb_2 (Crystal): Embedding 2.
		matcher (StructureMatcher | None): Pre-initialized StructureMatcher object.
		**kwargs: Additional keyword arguments for StructureMatcher, e.g., ltol, stol,
			angle_tol. If matcher is provided, these kwargs will be ignored.

	Returns:
		float: StructureMatcher-based distance (0.0 or 1.0).

	Note:
		d_smat does not allow pre-computation of embeddings.
	"""
	if matcher is None:
		matcher = StructureMatcher(**kwargs)
	return 0.0 if matcher.fit(emb_1, emb_2) else 1.0


def _d_comp(emb_1: TYPE_EMB_COMP, emb_2: TYPE_EMB_COMP, **kwargs) -> float:
	"""Compute the binary distance based on the match of compositions.

	Args:
		emb_1 (TYPE_EMB_COMP): Embedding 1.
		emb_2 (TYPE_EMB_COMP): Embedding 2.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		float: Composition distance (0.0 or 1.0).
	"""
	return 0.0 if emb_1 == emb_2 else 1.0


def _d_wyckoff(emb_1: TYPE_EMB_WYCKOFF, emb_2: TYPE_EMB_WYCKOFF, **kwargs) -> float:
	"""Compute the binary distance based on the match of Wyckoff representations.

	Args:
		emb_1 (TYPE_EMB_WYCKOFF): Embedding 1.
		emb_2 (TYPE_EMB_WYCKOFF): Embedding 2.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		float: Wyckoff distance (0.0 or 1.0).
	"""
	if isinstance(emb_1, _EmbeddingError) or isinstance(emb_2, _EmbeddingError):
		return float("nan")
	return 0.0 if emb_1 == emb_2 else 1.0


def _d_magpie(emb_1: TYPE_EMB_MAGPIE, emb_2: TYPE_EMB_MAGPIE, **kwargs) -> float:
	"""Compute the continuous distance using compositional Magpie fingerprints.

	Args:
		emb_1 (TYPE_EMB_MAGPIE): Embedding 1.
		emb_2 (TYPE_EMB_MAGPIE): Embedding 2.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		float: Magpie distance.

	References:
		- Ward et al., (2016). A general-purpose machine learning framework for
			predicting properties of inorganic materials. npj Computational Materials,
			2(1), 1-7. https://doi.org/10.1038/npjcompumats.2016.28
	"""
	return np.sqrt(np.sum((np.array(emb_1) - np.array(emb_2)) ** 2)).item()


def _d_pdd(emb_1: TYPE_EMB_PDD, emb_2: TYPE_EMB_PDD, **kwargs) -> float:
	"""Compute the continuous distance using pointwise distance distribution (PDD).

	Args:
		emb_1 (TYPE_EMB_PDD): Embedding 1.
		emb_2 (TYPE_EMB_PDD): Embedding 2.
		**kwargs: Additional arguments for amd.PDD_cdist.

	Returns:
		float: PDD distance.

	References:
		- Widdowson et al., (2022). Resolving the data ambiguity for periodic
			crystals. Advances in Neural Information Processing Systems, 35,
			24625--24638. https://openreview.net/forum?id=4wrB7Mo9_OQ
	"""
	if isinstance(emb_1, _EmbeddingError) or isinstance(emb_2, _EmbeddingError):
		return float("nan")
	return amd.PDD_cdist([emb_1], [emb_2], **kwargs)[0][0].item()


def _d_amd(emb_1: TYPE_EMB_AMD, emb_2: TYPE_EMB_AMD, **kwargs) -> float:
	"""Compute the continuous distance using average minimum distance (AMD).

	Args:
		emb_1 (TYPE_EMB_AMD): Embedding 1.
		emb_2 (TYPE_EMB_AMD): Embedding 2.
		**kwargs: Additional arguments for amd.AMD_cdist.

	Returns:
		float: AMD distance.

	References:
		- Widdson et al., (2022). Average Minimum Distances of periodic point sets -
			foundational invariants for mapping periodic crystals. MATCH
			Communications in Mathematical and in Computer Chemistry, 87(3), 529-559,
			https://doi.org/10.46793/match.87-3.529W
	"""
	if isinstance(emb_1, _EmbeddingError) or isinstance(emb_2, _EmbeddingError):
		return float("nan")
	return amd.AMD_cdist([emb_1], [emb_2], **kwargs)[0][0].item()


def _d_elmd(emb_1: TYPE_EMB_ELMD, emb_2: TYPE_EMB_ELMD, **kwargs) -> float:
	"""Compute continuous Element Movers Distance (ElMD).

	Args:
		emb_1 (TYPE_EMB_ELMD): Embedding 1.
		emb_2 (TYPE_EMB_ELMD): Embedding 2.
		**kwargs: Additional arguments for ElMD.elmd.

	Returns:
		float: ElMD distance.

	References:
		- Hargreaves et al., (2020). The Earth Mover's Distance as a Metric for the
			Space of Inorganic Compositions. Chemistry of Materials, 32(24),
			10610-10620.
	"""
	if "return_assignments" in kwargs and kwargs["return_assignments"] is True:
		warnings.warn(
			"return_assignments=True is not supported. Automatically set to False.",
			UserWarning,
			stacklevel=2,
		)
		kwargs["return_assignments"] = False
	return elmd(emb_1, emb_2, **kwargs)


def _set_n_processes(n_processes: int | None = None) -> int:
	"""Set the number of processes for multiprocessing.

	Args:
		n_processes (int | None): Number of processes. If None, use
			max(cpu_count() - 1, 1). Default is None.

	Returns:
		int: Number of processes to use.
	"""
	if n_processes is None:
		return max(cpu_count() - 1, 1)
	else:
		return max(min(n_processes, cpu_count() - 1), 1)


def _distance_matrix_template(
	embs_1: list,
	embs_2: list | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	d_func: Callable | None = None,
	initializer: Callable | None = None,
	initargs: tuple = (),
) -> np.ndarray:
	"""Template to compute the distance matrix between two sets of embeddings.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list): Embeddings
		embs_2 (list | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		d_func (Callable | None): Distance function.
		initializer (Callable | None): Initializer function for worker processes.
		initargs (tuple): Arguments for the initializer function.

	Returns:
		np.ndarray: Distance matrix.

	Raises:
		ValueError: If d_func is None.
	"""
	if d_func is None:
		raise ValueError("d_func must be provided.")
	given_two_sets = embs_2 is not None
	if given_two_sets:
		if multiprocessing:
			with Pool(
				processes=_set_n_processes(n_processes),
				initializer=initializer,
				initargs=initargs,
			) as pool:
				results = pool.starmap(
					d_func,
					((emb_i, emb_j) for emb_i in embs_1 for emb_j in embs_2),
				)
				d_mtx = np.array(results).reshape(len(embs_1), len(embs_2))
		else:
			d_mtx = np.zeros((len(embs_1), len(embs_2)))
			for i, emb_i in enumerate(embs_1):
				for j, emb_j in enumerate(embs_2):
					d_mtx[i, j] = d_func(emb_i, emb_j)
	else:
		if multiprocessing:
			with Pool(
				processes=_set_n_processes(n_processes),
				initializer=initializer,
				initargs=initargs,
			) as pool:
				results = pool.starmap(
					d_func,
					(
						(embs_1[i], embs_1[j])
						for i in range(len(embs_1))
						for j in range(i + 1)
					),
				)
				d_mtx = np.zeros((len(embs_1), len(embs_1)))
				for i in range(len(embs_1)):
					for j in range(i + 1):
						d_mtx[i, j] = results[i * (i + 1) // 2 + j]
						d_mtx[j, i] = d_mtx[i, j]
		else:
			d_mtx = np.zeros((len(embs_1), len(embs_1)))
			for i, emb_i in enumerate(embs_1):
				for j, emb_j in enumerate(embs_1[: i + 1]):
					d_mtx[i, j] = d_func(emb_i, emb_j)
					d_mtx[j, i] = d_mtx[i, j]
	return d_mtx


_smat_worker_args: tuple = ()


def _init_worker_distance_matrix_d_smat(
	prepared_1, prepared_2, matcher, subset, check_sites
):
	"""Keep prepared structures and a configured matcher in each worker.

	Args:
		prepared_1: Prepared source structures indexed by input position.
		prepared_2: Prepared target structures indexed by input position.
		matcher: Configured StructureMatcher.
		subset: Whether subset matches are allowed.
		check_sites: Whether unequal site counts rule out a match.
	"""
	global _smat_worker_args
	_smat_worker_args = (prepared_1, prepared_2, matcher, subset, check_sites)


def _worker_distance_matrix_d_smat(task):
	"""Evaluate one candidate row using worker-local structures and matcher.

	Args:
		task: Row index and candidate column indices.

	Returns:
		tuple: Row index, column indices, and distances.
	"""
	return _smat_row(task, *_smat_worker_args)


def _smat_row(task, prepared_1, prepared_2, matcher, subset, check_sites):
	"""Compute all candidate distances in a row without stopping at a match.

	Args:
		task: Row index and candidate column indices.
		prepared_1: Prepared source structures indexed by input position.
		prepared_2: Prepared target structures indexed by input position.
		matcher: Configured StructureMatcher.
		subset: Whether subset matches are allowed.
		check_sites: Whether unequal site counts rule out a match.

	Returns:
		tuple: Row index, column indices, and distances.
	"""
	i, columns = task
	original_1, reduced_1, key_1 = prepared_1[i]
	values = []
	for j in columns:
		original_2, reduced_2, key_2 = prepared_2[j]
		# Primitive reduction can change fractional occupancies across a hash boundary.
		if not subset and key_1 != key_2:
			value = _d_smat(original_1, original_2, matcher)
		elif check_sites and len(reduced_1) != len(reduced_2):
			value = 1.0
		else:
			value = (
				0.0
				if matcher.fit(reduced_1, reduced_2, skip_structure_reduction=True)
				else 1.0
			)
		values.append(value)
	return i, columns, values


def _smat_prepare(embs: list[Crystal], indices: set[int], **kwargs) -> dict:
	"""Reduce participating structures once, retaining originals for fallback.

	Args:
		embs: Input structures.
		indices: Positions participating in at least one candidate pair.
		**kwargs: StructureMatcher settings.

	Returns:
		dict: Input index to original structure, reduced structure, and composition key.
	"""
	comparator = kwargs.get("comparator") or SpeciesComparator()
	ignored_species = kwargs.get("ignored_species", ())
	prepared = {}
	for i in sorted(indices):
		structure = Structure.from_sites(embs[i])
		if ignored_species:
			structure.remove_species(ignored_species)
		structure = structure.get_reduced_structure(reduction_algo="niggli")
		if kwargs.get("primitive_cell", True):
			structure = structure.get_primitive_structure()
		prepared[i] = embs[i], structure, comparator.get_hash(structure.composition)
	return prepared


def _smat_composition_keys(embs: list[Crystal], **kwargs) -> list:
	"""Compute composition keys once for both candidate passes.

	Args:
		embs: Input structures.
		**kwargs: StructureMatcher settings.

	Returns:
		list: Comparator keys, or placeholders when subset matching bypasses pruning.
	"""
	if kwargs.get("allow_subset", False):
		return [None] * len(embs)
	comparator = kwargs.get("comparator") or SpeciesComparator()
	ignored_species = kwargs.get("ignored_species", ())
	keys = []
	for structure in embs:
		processed = Structure.from_sites(structure)
		if ignored_species:
			processed.remove_species(ignored_species)
		keys.append(comparator.get_hash(processed.composition))
	return keys


def _smat_candidate_rows(
	keys_1: list, keys_2: list | None, subset: bool
) -> Iterator[tuple[int, list[int]]]:
	"""Yield row indices and columns that can pass the matcher's composition check.

	Args:
		keys_1: Source composition keys.
		keys_2: Target composition keys, or None for a symmetric matrix.
		subset: Whether subset matching bypasses composition pruning.

	Yields:
		tuple[int, list[int]]: A row index and its candidate column indices.
	"""
	symmetric = keys_2 is None
	keys_2 = keys_1 if symmetric else keys_2
	if subset:
		for i in range(len(keys_1)):
			columns = list(range(i + 1 if symmetric else len(keys_2)))
			if columns:
				yield i, columns
		return

	# At the tolerance boundary, equal Compositions can have different element hashes.
	hash_safe = all(
		not isinstance(key, Composition)
		or all(abs(amount) > key.amount_tolerance for amount in key.values())
		for key in chain(keys_1, keys_2)
	)
	groups: dict[int, list[int]] = {}
	for j, key in enumerate(keys_2):
		groups.setdefault(hash(key) if hash_safe else 0, []).append(j)
	for i, key in enumerate(keys_1):
		# Composition equality has a tolerance and is not transitive: check each key.
		columns = [
			j
			for j in groups.get(hash(key) if hash_safe else 0, ())
			if (not symmetric or j <= i) and key == keys_2[j]
		]
		if columns:
			yield i, columns


def _distance_matrix_d_smat(
	embs_1: list[Crystal],
	embs_2: list[Crystal] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_smat.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[Crystal]): Embeddings
		embs_2 (list[Crystal] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Additional keyword arguments for StructureMatcher.

	Returns:
		np.ndarray: d_smat distance matrix.
	"""
	matcher = StructureMatcher(**kwargs)
	symmetric = embs_2 is None
	targets = embs_1 if symmetric else embs_2
	d_mtx = np.ones((len(embs_1), len(targets)))
	subset = kwargs.get("allow_subset", False)
	keys_1 = _smat_composition_keys(embs_1, **kwargs)
	keys_2 = None if symmetric else _smat_composition_keys(targets, **kwargs)
	indices_1 = set()
	indices_2 = indices_1 if symmetric else set()
	for i, columns in _smat_candidate_rows(keys_1, keys_2, subset):
		indices_1.add(i)
		indices_2.update(columns)
	if not indices_1:
		return d_mtx
	prepared_1 = _smat_prepare(embs_1, indices_1, **kwargs)
	prepared_2 = (
		prepared_1 if symmetric else _smat_prepare(targets, indices_2, **kwargs)
	)
	check_sites = not subset and not kwargs.get("attempt_supercell", False)
	worker_args = prepared_1, prepared_2, matcher, subset, check_sites
	rows = _smat_candidate_rows(keys_1, keys_2, subset)
	if multiprocessing:
		with Pool(
			processes=_set_n_processes(n_processes),
			initializer=_init_worker_distance_matrix_d_smat,
			initargs=worker_args,
		) as pool:
			for i, columns, values in pool.imap_unordered(
				_worker_distance_matrix_d_smat, rows
			):
				d_mtx[i, columns] = values
				if symmetric:
					d_mtx[columns, i] = values
	else:
		for task in rows:
			i, columns, values = _smat_row(task, *worker_args)
			d_mtx[i, columns] = values
			if symmetric:
				d_mtx[columns, i] = values
	return d_mtx


def _distance_matrix_d_comp(
	embs_1: list[TYPE_EMB_COMP],
	embs_2: list[TYPE_EMB_COMP] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_comp.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_COMP]): Embeddings
		embs_2 (list[TYPE_EMB_COMP] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		np.ndarray: d_comp distance matrix.
	"""
	return _distance_matrix_template(
		embs_1, embs_2, multiprocessing, n_processes, _d_comp
	)


def _distance_matrix_d_wyckoff(
	embs_1: list[TYPE_EMB_WYCKOFF],
	embs_2: list[TYPE_EMB_WYCKOFF] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_wyckoff.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_WYCKOFF]): Embeddings
		embs_2 (list[TYPE_EMB_WYCKOFF] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		np.ndarray: d_wyckoff distance matrix.
	"""
	return _distance_matrix_template(
		embs_1, embs_2, multiprocessing, n_processes, _d_wyckoff
	)


def _distance_matrix_d_magpie(
	embs_1: list[TYPE_EMB_MAGPIE],
	embs_2: list[TYPE_EMB_MAGPIE] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_magpie.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_MAGPIE]): Embeddings
		embs_2 (list[TYPE_EMB_MAGPIE] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Ignored; SciPy computes the matrix without a Python
			worker pool. Default is False.
		n_processes (int | None): Ignored; present for uniform interface.
		**kwargs: Ignored; present for uniform interface.

	Returns:
		np.ndarray: d_magpie distance matrix.
	"""
	return cdist(embs_1, embs_1 if embs_2 is None else embs_2, metric="euclidean")


def _distance_matrix_d_pdd(
	embs_1: list[TYPE_EMB_PDD],
	embs_2: list[TYPE_EMB_PDD] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_pdd.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_PDD]): Embeddings
		embs_2 (list[TYPE_EMB_PDD] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Additional arguments for amd.PDD_pdist and amd.PDD_cdist.

	Returns:
		np.ndarray: d_pdd distance matrix.
	"""
	if multiprocessing and "n_jobs" not in kwargs:
		kwargs["n_jobs"] = _set_n_processes(n_processes)

	given_two_sets = embs_2 is not None
	valids_1 = [emb for emb in embs_1 if not isinstance(emb, _EmbeddingError)]
	error_indices_1 = [
		i for i, emb in enumerate(embs_1) if isinstance(emb, _EmbeddingError)
	]
	if not given_two_sets:
		d_mtx = squareform(amd.PDD_pdist(valids_1, **kwargs))
		d_mtx = _insert_nan_rows_cols(d_mtx, error_indices_1, None)
	else:
		valids_2 = [emb for emb in embs_2 if not isinstance(emb, _EmbeddingError)]
		error_indices_2 = [
			i for i, emb in enumerate(embs_2) if isinstance(emb, _EmbeddingError)
		]
		d_mtx = amd.PDD_cdist(valids_1, valids_2, **kwargs)
		d_mtx = _insert_nan_rows_cols(d_mtx, error_indices_1, error_indices_2)
	return d_mtx


def _distance_matrix_d_amd(
	embs_1: list[TYPE_EMB_AMD],
	embs_2: list[TYPE_EMB_AMD] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_amd.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_AMD]): Embeddings
		embs_2 (list[TYPE_EMB_AMD] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Multiprocessing is
			not implemented for AMD; if True, a warning is issued and the computation
			proceeds without it. Default is False.
		n_processes (int | None): Ignored (multiprocessing not supported). Default is
			None.
		**kwargs: Additional arguments for amd.AMD_pdist and amd.AMD_cdist.

	Returns:
		np.ndarray: d_amd distance matrix.
	"""
	if multiprocessing:
		warnings.warn(
			"Multiprocessing is not implemented for _distance_matrix_d_amd. "
			"Proceeding without multiprocessing.",
			stacklevel=2,
		)

	given_two_sets = embs_2 is not None
	valids_1 = [emb for emb in embs_1 if not isinstance(emb, _EmbeddingError)]
	error_indices_1 = [
		i for i, emb in enumerate(embs_1) if isinstance(emb, _EmbeddingError)
	]
	if not given_two_sets:
		d_mtx = squareform(amd.AMD_pdist(valids_1, **kwargs))
		d_mtx = _insert_nan_rows_cols(d_mtx, error_indices_1, None)
	else:
		valids_2 = [emb for emb in embs_2 if not isinstance(emb, _EmbeddingError)]
		error_indices_2 = [
			i for i, emb in enumerate(embs_2) if isinstance(emb, _EmbeddingError)
		]
		d_mtx = amd.AMD_cdist(valids_1, valids_2, **kwargs)
		d_mtx = _insert_nan_rows_cols(d_mtx, error_indices_1, error_indices_2)
	return d_mtx


def _distance_matrix_d_elmd(
	embs_1: list[TYPE_EMB_ELMD],
	embs_2: list[TYPE_EMB_ELMD] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_elmd.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_ELMD]): Embeddings
		embs_2 (list[TYPE_EMB_ELMD] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Additional arguments for ElMD.elmd.

	Returns:
		np.ndarray: d_elmd distance matrix.
	"""
	return _distance_matrix_template(
		embs_1,
		embs_2,
		multiprocessing,
		n_processes,
		functools.partial(_d_elmd, **kwargs),
	)


def _compute_embedding_worker(xtal: Crystal, distance: str, **kwargs) -> TYPE_EMB_ALL:
	"""Worker function for computing embedding in multiprocessing.

	Compute the embedding for a crystal based on the specified distance metric.

	Args:
		xtal (Crystal): A Crystal object.
		distance (str): The distance metric to use.
		**kwargs: Additional arguments for embedding methods if needed.

	Returns:
		TYPE_EMB_ALL | _EmbeddingError: The embedding, or an _EmbeddingError
			wrapping the exception if embedding computation failed.
	"""
	try:
		return xtal.get_embedding(distance, **kwargs)
	except Exception as e:
		return _EmbeddingError(e)


def _compute_embeddings(
	distance: str,
	xtals: Crystal | list[Crystal],
	multiprocessing: bool,
	n_processes: int | None = None,
	**kwargs,
) -> TYPE_EMB_ALL | list[TYPE_EMB_ALL]:
	"""Compute embedding(s) for given crystal(s) based on the specified distance metric.

	Args:
		distance (str): The distance metric to use.
		xtals (Crystal | list[Crystal]): A Crystal object or a list of Crystal objects.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		**kwargs: Additional arguments for embedding methods if needed.

	Returns:
		TYPE_EMB_ALL | list[TYPE_EMB_ALL]: A single embedding or a list of embeddings.
	"""
	single = False
	if isinstance(xtals, Crystal):
		xtals = [xtals]
		single = True

	if multiprocessing:
		_compute_embedding_worker_partial = functools.partial(
			_compute_embedding_worker, distance=distance, **kwargs
		)

		with Pool(processes=_set_n_processes(n_processes)) as pool:
			embs = pool.map(
				_compute_embedding_worker_partial,
				xtals,
			)
	else:
		embs = [_compute_embedding_worker(xtal, distance, **kwargs) for xtal in xtals]
	if single:
		return embs[0]
	else:
		return embs


# Registry: metric name → pairwise distance function
_DIST_FUNCS: dict[str, Callable] = {
	"smat": _d_smat,
	"comp": _d_comp,
	"wyckoff": _d_wyckoff,
	"magpie": _d_magpie,
	"pdd": _d_pdd,
	"amd": _d_amd,
	"elmd": _d_elmd,
}

# Registry: metric name → distance matrix function
_DIST_MTX_FUNCS: dict[str, Callable] = {
	"smat": _distance_matrix_d_smat,
	"comp": _distance_matrix_d_comp,
	"wyckoff": _distance_matrix_d_wyckoff,
	"magpie": _distance_matrix_d_magpie,
	"pdd": _distance_matrix_d_pdd,
	"amd": _distance_matrix_d_amd,
	"elmd": _distance_matrix_d_elmd,
}


@overload
def distance(
	distance: str,
	xtal_1: Structure | Crystal | TYPE_EMB_ALL,
	xtal_2: Structure | Crystal | TYPE_EMB_ALL,
	normalize: bool = ...,
	verbose: Literal[False] = ...,
	**kwargs,
) -> float: ...


@overload
def distance(
	distance: str,
	xtal_1: Structure | Crystal | TYPE_EMB_ALL,
	xtal_2: Structure | Crystal | TYPE_EMB_ALL,
	normalize: bool = ...,
	verbose: Literal[True] = ...,
	**kwargs,
) -> tuple[float, TYPE_EMB_ALL, TYPE_EMB_ALL]: ...


def distance(
	distance: str,
	xtal_1: Structure | Crystal | TYPE_EMB_ALL,
	xtal_2: Structure | Crystal | TYPE_EMB_ALL,
	normalize: bool = True,
	verbose: bool = False,
	**kwargs,
) -> float | tuple[float, TYPE_EMB_ALL, TYPE_EMB_ALL]:
	"""Compute the distance between two crystals.

	Args:
		distance (str): The distance metric to use. Currently supported metrics are
			listed in SUPPORTED_DISTANCES in constants.py. For more detailed information
			about each distance metric, please refer to the `tutorial notebook`_.
		xtal_1 (Structure | Crystal | TYPE_EMB_ALL): pymatgen Structure or
			Crystal or an embedding.
		xtal_2 (Structure | Crystal | TYPE_EMB_ALL): pymatgen Structure or
			Crystal or an embedding.
		normalize (bool): Whether to normalize the distance d to [0, 1] by using d'
			= d / (1 + d). This argument is only considered when d is a continuous
			distance that is not normalized to [0, 1]. Such distances are listed in
			CONTINUOUS_UNNORMALIZED_DISTANCES in constants.py. Default is True.
		verbose (bool): Whether to return intermediate embeddings. Default is False.
		**kwargs: Additional keyword arguments for specific distance metrics. It can
			contain two keys: "args_emb" and "args_dist". The value of "args_emb" is a
			dict of arguments for the calculation of embeddings, and the value of
			"args_dist" is a dict of arguments for the calculation of distance between
			the embeddings. If embeddings are pre-computed and provided as inputs, the
			"args_emb" will be ignored.


	Returns:
		float |  tuple[np.ndarray, TYPE_EMB_ALL, TYPE_EMB_ALL, dict[str, float]]:
		Distance between crystals. If verbose is True, also returns the embeddings and
		the computing time.

	Raises:
		ValueError: If an unsupported distance metric is provided.

	.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
	"""
	# conversions from Structure to Crystal
	xtal_1 = _to_crystal(xtal_1)
	xtal_2 = _to_crystal(xtal_2)

	# compute embeddings
	if distance not in DIST_WO_EMB and isinstance(xtal_1, Crystal):
		emb_1 = _compute_embeddings(
			distance, xtal_1, False, **(kwargs.get("args_emb", {}))
		)
	else:
		emb_1 = xtal_1
	if distance not in DIST_WO_EMB and isinstance(xtal_2, Crystal):
		emb_2 = _compute_embeddings(
			distance, xtal_2, False, **(kwargs.get("args_emb", {}))
		)
	else:
		emb_2 = xtal_2

	# compute distance
	if distance == "elmd+amd":
		d_elmd_unnorm = _d_elmd(
			emb_1[0], emb_2[0], **(kwargs.get("args_dist", {}).get("elmd", {}))
		)
		d_amd_unnorm = _d_amd(
			emb_1[1], emb_2[1], **(kwargs.get("args_dist", {}).get("amd", {}))
		)
		d = _blend_elmd_amd_distance(
			d_elmd_unnorm, d_amd_unnorm, kwargs.get("args_dist", {})
		)
	elif distance in _DIST_FUNCS:
		d = _DIST_FUNCS[distance](emb_1, emb_2, **(kwargs.get("args_dist", {})))
	else:
		raise ValueError(f"Unsupported distance metric: {distance}")

	# normalize distance if needed
	if distance in CONTINUOUS_UNNORMALIZED_DISTANCES and normalize:
		d = _normalize(d)

	# return results
	if not verbose:
		return d
	else:
		return d, emb_1, emb_2


TYPE_D_MTX_RETURN = (
	np.ndarray
	| tuple[np.ndarray, list[TYPE_EMB_ALL], dict[str, float]]
	| tuple[
		np.ndarray,
		list[TYPE_EMB_ALL],
		list[TYPE_EMB_ALL],
		dict[str, float],
	]
)  #: Return type for distance_matrix function.


def distance_matrix(
	distance: str,
	xtals_1: list[Structure | Crystal | TYPE_EMB_ALL],
	xtals_2: list[Structure | Crystal | TYPE_EMB_ALL] | None = None,
	normalize: bool = True,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	verbose: bool = False,
	**kwargs,
) -> TYPE_D_MTX_RETURN:
	"""Compute the distance matrix between two sets of crystals.

	If xtals_2 is None, compute the distance matrix within xtals_1.

	Args:
		distance (str): The distance metric to use. Currently supported metrics are
			listed in SUPPORTED_DISTANCES in constants.py. For more detailed information
			about each distance metric, please refer to the `tutorial notebook`_.
		xtals_1 (list[Structure | Crystal | TYPE_EMB_ALL]): A list of pymatgen
			Structures or Crystals or embeddings.
		xtals_2 (list[Structure | Crystal | TYPE_EMB_ALL] | None): A list of
			pymatgen Structures or Crystals or embeddings, or None. Default is None.
		normalize (bool): Whether to normalize the distances d to [0, 1] by using d'
			= d / (1 + d). This argument is only considered when d is a continuous
			distance that is not normalized to [0, 1]. Such distances are listed in
			CONTINUOUS_UNNORMALIZED_DISTANCES in constants.py. Default is True.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.
		verbose (bool): Whether to return embeddings and the computing time. Default is
			False.
		**kwargs: Additional keyword arguments for specific distance metrics. It can
			contain two keys: "args_emb" and "args_dist". The value of "args_emb" is a
			dict of arguments for the calculation of embeddings, and the value of
			"args_dist" is a dict of arguments for the calculation of distance matrix
			using the embeddings. If embeddings are pre-computed and provided as inputs,
			the "args_emb" will be ignored.

	Returns:
		TYPE_D_MTX_RETURN: Distance matrix, the embeddings of xtals_1 (and xtals_2 if
		xtals_2 is not None) and the computing time.

	Notes:
		For "smat", the dense matrix uses 0 for matches and 1 for mismatches.
		Composition-incompatible pairs are skipped using the configured comparator
		and ignored species; subset matching compares all pairs. Serial and parallel
		calculations both honor args_dist and reuse a matcher. Parallel tasks contain
		candidate row indices rather than a materialized list of all structure pairs.
		Participating structures undergo Niggli and optional primitive reduction
		once per input position in the parent process, after removing ignored species.
		Workers reuse the reductions; volume scaling and supercell matching remain
		pairwise. Unequal reduced site counts rule out matches unless subset or
		supercell matching is enabled. Pairs whose reduced compositions become
		incompatible fall back to normal matching of the original structures.
		Input structures and returned embeddings are not replaced by reductions.

		For "magpie", SciPy computes the dense Euclidean distance matrix without a
		Python worker pool. Multiprocessing settings still apply to embedding
		computation when structures are provided, but not to the matrix calculation.
		Magpie normalization updates the final matrix in place, one row at a time,
		to avoid additional full-size arrays. Dense output requires 8 bytes
		per pair: about 2.24 GiB for a 10,000-by-30,000 float64 matrix, excluding
		embeddings and other evaluation data.

	Raises:
		ValueError: If an unsupported distance metric is provided.

	.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
	"""
	given_two_sets = xtals_2 is not None
	times = {}

	# conversions from Structure to Crystal
	xtals_1 = _to_crystal_list(xtals_1)
	if given_two_sets:
		xtals_2 = _to_crystal_list(xtals_2)

	# compute embeddings
	if distance not in DIST_WO_EMB and isinstance(xtals_1[0], Crystal):
		emb_1_start = time.time()
		embs_1 = _compute_embeddings(
			distance,
			xtals_1,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_emb", {})),
		)
		emb_1_end = time.time()
		times["emb_1"] = emb_1_end - emb_1_start
	else:
		embs_1 = xtals_1
		times["emb_1"] = 0.0
	if given_two_sets:
		if distance not in DIST_WO_EMB and isinstance(xtals_2[0], Crystal):
			emb_2_start = time.time()
			embs_2 = _compute_embeddings(
				distance,
				xtals_2,
				multiprocessing,
				n_processes,
				**(kwargs.get("args_emb", {})),
			)
			emb_2_end = time.time()
			times["emb_2"] = emb_2_end - emb_2_start
		else:
			embs_2 = xtals_2
			times["emb_2"] = 0.0
	else:
		embs_2 = None

	# compute distances
	d_mtx_start = time.time()
	if distance == "elmd+amd":
		d_mtx_elmd_unnorm = _distance_matrix_d_elmd(
			[emb[0] for emb in embs_1],
			[emb[0] for emb in embs_2] if embs_2 is not None else None,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_dist", {}).get("elmd", {})),
		)
		d_mtx_amd_unnorm = _distance_matrix_d_amd(
			[emb[1] for emb in embs_1],
			[emb[1] for emb in embs_2] if embs_2 is not None else None,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_dist", {}).get("amd", {})),
		)
		d_mtx = _blend_elmd_amd_distance(
			d_mtx_elmd_unnorm, d_mtx_amd_unnorm, kwargs.get("args_dist", {})
		)
	elif distance in _DIST_MTX_FUNCS:
		d_mtx = _DIST_MTX_FUNCS[distance](
			embs_1,
			embs_2,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_dist", {})),
		)
	else:
		raise ValueError(f"Unsupported distance metric: {distance}")
	d_mtx_end = time.time()
	times["d_mtx"] = d_mtx_end - d_mtx_start

	# normalize distances if needed
	if distance in CONTINUOUS_UNNORMALIZED_DISTANCES and normalize:
		if distance == "magpie":
			for row in d_mtx:
				row /= 1 + row
		else:
			d_mtx = _normalize(d_mtx)

	# return results
	if not verbose:
		return d_mtx
	elif given_two_sets:
		return d_mtx, embs_1, embs_2, times
	else:
		return d_mtx, embs_1, times
