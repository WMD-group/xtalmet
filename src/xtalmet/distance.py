"""This module offers a range of distance functions for crystals."""

import time

import amd
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.spatial.distance import squareform

from .constants import (
	DIST_WO_EMB,
	TYPE_EMB_ALL,
	TYPE_EMB_AMD,
	TYPE_EMB_COMP,
	TYPE_EMB_MAGPIE,
	TYPE_EMB_PDD,
	TYPE_EMB_WYCKOFF,
)
from .crystal import Crystal


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


def _d_comp(emb_1: TYPE_EMB_COMP, emb_2: TYPE_EMB_COMP) -> float:
	"""Compute the binary distance based on the match of compositions.

	Args:
		emb_1 (TYPE_EMB_COMP): Embedding 1.
		emb_2 (TYPE_EMB_COMP): Embedding 2.

	Returns:
		float: Composition distance (0.0 or 1.0).
	"""
	return 0.0 if emb_1 == emb_2 else 1.0


def _d_wyckoff(emb_1: TYPE_EMB_WYCKOFF, emb_2: TYPE_EMB_WYCKOFF) -> float:
	"""Compute the binary distance based on the match of Wyckoff representations.

	Args:
		emb_1 (TYPE_EMB_WYCKOFF): Embedding 1.
		emb_2 (TYPE_EMB_WYCKOFF): Embedding 2.

	Returns:
		float: Wyckoff distance (0.0 or 1.0).
	"""
	if isinstance(emb_1, Exception) or isinstance(emb_2, Exception):
		return float("nan")
	return 0.0 if emb_1 == emb_2 else 1.0


def _d_magpie(emb_1: TYPE_EMB_MAGPIE, emb_2: TYPE_EMB_MAGPIE) -> float:
	"""Compute the continuous distance using compositional Magpie fingerprints.

	Args:
		emb_1 (TYPE_EMB_MAGPIE): Embedding 1.
		emb_2 (TYPE_EMB_MAGPIE): Embedding 2.

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
	if isinstance(emb_1, Exception) or isinstance(emb_2, Exception):
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
	if isinstance(emb_1, Exception) or isinstance(emb_2, Exception):
		return float("nan")
	return amd.AMD_cdist([emb_1], [emb_2], **kwargs)[0][0].item()


def _distance_matrix_d_smat(
	embs_1: list[Crystal], embs_2: list[Crystal] | None = None, **kwargs
):
	"""Compute the distance matrix between two sets of embeddings based on d_smat.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[Crystal]): Embeddings
		embs_2 (list[Crystal] | None): Embeddings or None. Default is None.
		**kwargs: Additional keyword arguments for StructureMatcher.

	Returns:
		np.ndarray: d_smat distance matrix.
	"""
	given_two_sets = embs_2 is not None
	d_mtx = np.ones((len(embs_1), len(embs_2) if given_two_sets else len(embs_1)))
	matcher = StructureMatcher(**kwargs)
	if given_two_sets:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_2):
				d_mtx[i, j] = _d_smat(emb_i, emb_j, matcher=matcher)
	else:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_1[: i + 1]):
				d_mtx[i, j] = _d_smat(emb_i, emb_j, matcher=matcher)
				d_mtx[j, i] = d_mtx[i, j]
	return d_mtx


def _distance_matrix_d_comp(
	embs_1: list[TYPE_EMB_COMP], embs_2: list[TYPE_EMB_COMP] | None = None
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_comp.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_COMP]): Embeddings
		embs_2 (list[TYPE_EMB_COMP] | None): Embeddings or None. Default is None.

	Returns:
		np.ndarray: d_comp distance matrix.
	"""
	given_two_sets = embs_2 is not None
	d_mtx = np.ones((len(embs_1), len(embs_2) if given_two_sets else len(embs_1)))
	if given_two_sets:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_2):
				d_mtx[i, j] = _d_comp(emb_i, emb_j)
	else:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_1[: i + 1]):
				d_mtx[i, j] = _d_comp(emb_i, emb_j)
				d_mtx[j, i] = d_mtx[i, j]
	return d_mtx


def _distance_matrix_d_wyckoff(
	embs_1: list[TYPE_EMB_WYCKOFF], embs_2: list[TYPE_EMB_WYCKOFF] | None = None
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_wyckoff.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_WYCKOFF]): Embeddings
		embs_2 (list[TYPE_EMB_WYCKOFF] | None): Embeddings or None. Default is None.

	Returns:
		np.ndarray: d_wyckoff distance matrix.
	"""
	given_two_sets = embs_2 is not None
	d_mtx = np.ones((len(embs_1), len(embs_2) if given_two_sets else len(embs_1)))
	if given_two_sets:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_2):
				d_mtx[i, j] = _d_wyckoff(emb_i, emb_j)
	else:
		for i, emb_i in enumerate(embs_1):
			for j, emb_j in enumerate(embs_1[: i + 1]):
				d_mtx[i, j] = _d_wyckoff(emb_i, emb_j)
				d_mtx[j, i] = d_mtx[i, j]
	return d_mtx


def _distance_matrix_d_magpie(
	embs_1: list[TYPE_EMB_MAGPIE], embs_2: list[TYPE_EMB_MAGPIE] | None = None
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_magpie.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_MAGPIE]): Embeddings
		embs_2 (list[TYPE_EMB_MAGPIE] | None): Embeddings or None. Default is None.

	Returns:
		np.ndarray: d_magpie distance matrix.
	"""
	if embs_2 is None:
		embs_2 = embs_1
	d_mtx = np.zeros((len(embs_1), len(embs_2)))
	embs_1 = np.array(embs_1)
	embs_2 = np.array(embs_2)
	for i, emb in enumerate(embs_1):
		d_sq = (emb[np.newaxis, :] - embs_2) ** 2
		d_euclidean = np.sqrt(np.sum(d_sq, axis=1))
		d_mtx[i, :] = d_euclidean
	return d_mtx


def _distance_matrix_d_pdd(
	embs_1: list[TYPE_EMB_PDD], embs_2: list[TYPE_EMB_PDD] | None = None, **kwargs
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_pdd.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_PDD]): Embeddings
		embs_2 (list[TYPE_EMB_PDD] | None): Embeddings or None. Default is None.
		**kwargs: Additional arguments for amd.PDD_pdist and amd.PDD_cdist.

	Returns:
		np.ndarray: d_pdd distance matrix.
	"""
	given_two_sets = embs_2 is not None
	valids_1 = [emb for emb in embs_1 if not isinstance(emb, Exception)]
	error_indices_1 = [i for i, emb in enumerate(embs_1) if isinstance(emb, Exception)]
	if not given_two_sets:
		d_mtx = squareform(amd.PDD_pdist(valids_1, **kwargs))
	else:
		valids_2 = [emb for emb in embs_2 if not isinstance(emb, Exception)]
		error_indices_2 = [
			i for i, emb in enumerate(embs_2) if isinstance(emb, Exception)
		]
		d_mtx = amd.PDD_cdist(valids_1, valids_2, **kwargs)
	# insert NaN for error embeddings
	for i in error_indices_1:
		d_mtx = np.insert(d_mtx, i, float("nan"), axis=0)
	if given_two_sets:
		for j in error_indices_2:
			d_mtx = np.insert(d_mtx, j, float("nan"), axis=1)
	else:
		for j in error_indices_1:
			d_mtx = np.insert(d_mtx, j, float("nan"), axis=1)
	return d_mtx


def _distance_matrix_d_amd(
	embs_1: list[TYPE_EMB_AMD], embs_2: list[TYPE_EMB_AMD] | None = None, **kwargs
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_amd.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_AMD]): Embeddings
		embs_2 (list[TYPE_EMB_AMD] | None): Embeddings or None. Default is None.
		**kwargs: Additional arguments for amd.AMD_pdist and amd.AMD_cdist.

	Returns:
		np.ndarray: d_amd distance matrix.
	"""
	given_two_sets = embs_2 is not None
	valids_1 = [emb for emb in embs_1 if not isinstance(emb, Exception)]
	error_indices_1 = [i for i, emb in enumerate(embs_1) if isinstance(emb, Exception)]
	if not given_two_sets:
		d_mtx = squareform(amd.AMD_pdist(valids_1, **kwargs))
	else:
		valids_2 = [emb for emb in embs_2 if not isinstance(emb, Exception)]
		error_indices_2 = [
			i for i, emb in enumerate(embs_2) if isinstance(emb, Exception)
		]
		d_mtx = amd.AMD_cdist(valids_1, valids_2, **kwargs)
	# insert NaN for error embeddings
	for i in error_indices_1:
		d_mtx = np.insert(d_mtx, i, float("nan"), axis=0)
	if given_two_sets:
		for j in error_indices_2:
			d_mtx = np.insert(d_mtx, j, float("nan"), axis=1)
	else:
		for j in error_indices_1:
			d_mtx = np.insert(d_mtx, j, float("nan"), axis=1)
	return d_mtx


def _compute_embeddings(
	distance: str, xtals: Crystal | list[Crystal], **kwargs
) -> TYPE_EMB_ALL | list[TYPE_EMB_ALL]:
	"""Compute embedding(s) for given crystal(s) based on the specified distance metric.

	Args:
		distance (str): The distance metric to use.
		xtals (Crystal | list[Crystal]): A Crystal object or a list of Crystal objects.
		**kwargs: Additional arguments for embedding methods if needed.

	Returns:
		TYPE_EMB_ALL | list[TYPE_EMB_ALL]: A list of embeddings.
	"""
	if isinstance(xtals, Crystal):
		xtals = [xtals]
	embs = []
	for xtal in xtals:
		try:
			embs.append(xtal.get_embedding(distance, **kwargs))
		except Exception as e:
			embs.append(e)
	if len(embs) == 1:
		return embs[0]
	else:
		return embs


def distance(
	distance: str,
	xtal_1: Structure | Crystal | TYPE_EMB_ALL,
	xtal_2: Structure | Crystal | TYPE_EMB_ALL,
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
		verbose (bool): Whether to return intermediate embeddings. Default is False.
		**kwargs: Additional keyword arguments for specific distance metrics. It can
			contain two keys: "args_emb" and "args_dist". The value of "args_emb" is a
			dict of arguments for the calculation of embeddings, and the value of
			"args_dist" is a dict of arguments for the calculation of distance between
			the embeddings. If embeddings are pre-computed and provided as inputs, the
			"args_emb" will be ignored.


	Returns:
		float |  tuple[np.ndarray, TYPE_EMB_ALL, TYPE_EMB_ALL,
			dict[str, float]]: Distance between crystals. If verbose is True, also
			returns the embeddings and the computing time.

	.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
	"""
	# conversions from Structure to Crystal
	xtal_1 = Crystal.from_Structure(xtal_1) if isinstance(xtal_1, Structure) else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if isinstance(xtal_2, Structure) else xtal_2

	# compute embeddings
	if distance not in DIST_WO_EMB and isinstance(xtal_1, Crystal):
		emb_1 = _compute_embeddings(distance, xtal_1, **(kwargs.get("args_emb", {})))
	else:
		emb_1 = xtal_1
	if distance not in DIST_WO_EMB and isinstance(xtal_2, Crystal):
		emb_2 = _compute_embeddings(distance, xtal_2, **(kwargs.get("args_emb", {})))
	else:
		emb_2 = xtal_2

	# compute distance
	if distance == "smat":
		d = _d_smat(emb_1, emb_2, **(kwargs.get("args_dist", {})))
	elif distance == "comp":
		d = _d_comp(emb_1, emb_2)
	elif distance == "wyckoff":
		d = _d_wyckoff(emb_1, emb_2)
	elif distance == "magpie":
		d = _d_magpie(emb_1, emb_2)
	elif distance == "pdd":
		d = _d_pdd(emb_1, emb_2, **(kwargs.get("args_dist", {})))
	elif distance == "amd":
		d = _d_amd(emb_1, emb_2, **(kwargs.get("args_dist", {})))
	else:
		raise ValueError(f"Unsupported distance metric: {distance}")

	# return results
	if not verbose:
		return d
	else:
		return d, emb_1, emb_2


def distance_matrix(
	distance: str,
	xtals_1: list[Structure | Crystal | TYPE_EMB_ALL],
	xtals_2: list[Structure | Crystal | TYPE_EMB_ALL] | None = None,
	verbose: bool = False,
	**kwargs,
) -> (
	np.ndarray
	| tuple[np.ndarray, list[TYPE_EMB_ALL], dict[str, float]]
	| tuple[
		np.ndarray,
		list[TYPE_EMB_ALL],
		list[TYPE_EMB_ALL],
		dict[str, float],
	]
):
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
		verbose (bool): Whether to return embeddings and the computing time. Default is
			False.
		**kwargs: Additional keyword arguments for specific distance metrics. It can
			contain two keys: "args_emb" and "args_dist". The value of "args_emb" is a
			dict of arguments for the calculation of embeddings, and the value of
			"args_dist" is a dict of arguments for the calculation of distance matrix
			using the embeddings.

	Returns:
		np.ndarray | tuple[np.ndarray, list[TYPE_EMB_ALL], dict[str, float]] |
			tuple[np.ndarray, list[TYPE_EMB_ALL], list[TYPE_EMB_ALL],
			dict[str, float]]: Distance matrix, the embeddings of xtals_1 (and xtals_2
			if xtals_2 is not None) and the computing time.

	.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
	"""
	given_two_sets = xtals_2 is not None
	times = {}

	# conversions from Structure to Crystal
	xtals_1 = [
		Crystal.from_Structure(x) if isinstance(x, Structure) else x for x in xtals_1
	]
	if given_two_sets:
		xtals_2 = [
			Crystal.from_Structure(x) if isinstance(x, Structure) else x
			for x in xtals_2
		]

	# compute embeddings
	if distance not in DIST_WO_EMB and isinstance(xtals_1[0], Crystal):
		emb_1_start = time.time()
		embs_1 = _compute_embeddings(distance, xtals_1, **(kwargs.get("args_emb", {})))
		emb_1_end = time.time()
		times["emb_1"] = emb_1_end - emb_1_start
	else:
		embs_1 = xtals_1
		times["emb_1"] = 0.0
	if given_two_sets:
		if distance not in DIST_WO_EMB and isinstance(xtals_2[0], Crystal):
			emb_2_start = time.time()
			embs_2 = _compute_embeddings(
				distance, xtals_2, **(kwargs.get("args_emb", {}))
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
	if distance == "smat":
		d_mtx = _distance_matrix_d_smat(embs_1, embs_2, **(kwargs.get("args_dist", {})))
	elif distance == "comp":
		d_mtx = _distance_matrix_d_comp(embs_1, embs_2)
	elif distance == "wyckoff":
		d_mtx = _distance_matrix_d_wyckoff(embs_1, embs_2)
	elif distance == "magpie":
		d_mtx = _distance_matrix_d_magpie(embs_1, embs_2)
	elif distance == "pdd":
		d_mtx = _distance_matrix_d_pdd(embs_1, embs_2, **(kwargs.get("args_dist", {})))
	elif distance == "amd":
		d_mtx = _distance_matrix_d_amd(embs_1, embs_2, **(kwargs.get("args_dist", {})))
	else:
		raise ValueError(f"Unsupported distance metric: {distance}")
	d_mtx_end = time.time()
	times["d_mtx"] = d_mtx_end - d_mtx_start

	# return results
	if not verbose:
		return d_mtx
	elif given_two_sets:
		return d_mtx, embs_1, embs_2, times
	else:
		return d_mtx, embs_1, times
