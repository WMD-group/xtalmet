"""This module offers a range of distance functions for crystals."""

import functools
import time
import warnings
from collections.abc import Callable
from multiprocessing import Pool, cpu_count

import amd
import numpy as np
from ElMD import elmd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.spatial.distance import squareform

from .constants import (
	CONTINUOUS_UNNORMALIZED_DISTANCES,
	DIST_WO_EMB,
	TYPE_EMB_ALL,
	TYPE_EMB_AMD,
	TYPE_EMB_COMP,
	TYPE_EMB_ELMD,
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


def _d_elmd(emb_1: TYPE_EMB_ELMD, emb_2: TYPE_EMB_ELMD, **kwargs) -> float:
	"""Compute continuous Element Movers Distance (ElMD).

	Args:
		emb_1 (TYPE_EMB_ELMD): Embedding 1.
		emb_2 (TYPE_EMB_ELMD): Embedding 2.
		**kwargs: Additional arguments for ElMD.elmd.

	Returns:
		float: ElMD distance.

	References:
		- Hargreaves et al., (2020). The Earth Mover’s Distance as a Metric for the
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
	"""
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


_global_matcher: StructureMatcher | None = None


def _init_worker_distance_matrix_d_smat(kwargs_dict: dict):
	"""Initialize global StructureMatcher for worker processes.

	Args:
		kwargs_dict (dict): Additional keyword arguments for StructureMatcher.
	"""
	global _global_matcher
	_global_matcher = StructureMatcher(**kwargs_dict)


def _worker_distance_matrix_d_smat(emb_i, emb_j):
	"""Worker function for computing d_smat in multiprocessing.

	Args:
		emb_i: Embedding i.
		emb_j: Embedding j.

	Returns:
		float: d_smat distance between emb_i and emb_j.
	"""
	global _global_matcher
	return _d_smat(emb_i, emb_j, matcher=_global_matcher)


def _distance_matrix_d_smat(
	embs_1: list[Crystal],
	embs_2: list[Crystal] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
	**kwargs,
):
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
	return _distance_matrix_template(
		embs_1,
		embs_2,
		multiprocessing,
		n_processes,
		_worker_distance_matrix_d_smat,
		_init_worker_distance_matrix_d_smat,
		(kwargs,),
	)


def _distance_matrix_d_comp(
	embs_1: list[TYPE_EMB_COMP],
	embs_2: list[TYPE_EMB_COMP] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_comp.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_COMP]): Embeddings
		embs_2 (list[TYPE_EMB_COMP] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.

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
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_wyckoff.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_WYCKOFF]): Embeddings
		embs_2 (list[TYPE_EMB_WYCKOFF] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.

	Returns:
		np.ndarray: d_wyckoff distance matrix.
	"""
	return _distance_matrix_template(
		embs_1, embs_2, multiprocessing, n_processes, _d_wyckoff
	)


_embs_2: np.ndarray | None = None


def _init_worker_distance_matrix_d_magpie(embs_2: np.ndarray):
	"""Initialize global variables for worker processes.

	Args:
		embs_2 (np.ndarray): 2D array of embeddings.
	"""
	global _embs_2
	_embs_2 = embs_2


def _worker_distance_matrix_d_magpie(emb: np.ndarray) -> np.ndarray:
	"""Worker function for computing a row of the distance matrix in multiprocessing.

	Compute a row of the distance matrix based on d_magpie.

	Args:
		emb (np.ndarray): 1D array of a single embedding.

	Returns:
		np.ndarray: A row of the d_magpie distance matrix.
	"""
	global _embs_2
	d_sq = (emb[np.newaxis, :] - _embs_2) ** 2
	d_euclidean = np.sqrt(np.sum(d_sq, axis=1))
	return d_euclidean


def _distance_matrix_d_magpie(
	embs_1: list[TYPE_EMB_MAGPIE],
	embs_2: list[TYPE_EMB_MAGPIE] | None = None,
	multiprocessing: bool = False,
	n_processes: int | None = None,
) -> np.ndarray:
	"""Compute the distance matrix between two sets of embeddings based on d_magpie.

	If embs_2 is None, compute the distance matrix within embs_1.

	Args:
		embs_1 (list[TYPE_EMB_MAGPIE]): Embeddings
		embs_2 (list[TYPE_EMB_MAGPIE] | None): Embeddings or None. Default is None.
		multiprocessing (bool): Whether to use multiprocessing. Default is False.
		n_processes (int | None): Maximum number of processes for multiprocessing. If
			multiprocessing is False, this argument will be ignored. Default is None.

	Returns:
		np.ndarray: d_magpie distance matrix.
	"""
	if embs_2 is None:
		embs_2 = embs_1
	d_mtx = np.zeros((len(embs_1), len(embs_2)))
	embs_1 = np.array(embs_1)
	embs_2 = np.array(embs_2)
	if multiprocessing:
		with Pool(
			processes=_set_n_processes(n_processes),
			initializer=_init_worker_distance_matrix_d_magpie,
			initargs=(embs_2,),
		) as pool:
			results = pool.map(
				_worker_distance_matrix_d_magpie,
				embs_1,
			)
			for i, row in enumerate(results):
				d_mtx[i, :] = row
	else:
		for i, emb in enumerate(embs_1):
			d_sq = (emb[np.newaxis, :] - embs_2) ** 2
			d_euclidean = np.sqrt(np.sum(d_sq, axis=1))
			d_mtx[i, :] = d_euclidean
	return d_mtx


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
	embs_1: list[TYPE_EMB_AMD],
	embs_2: list[TYPE_EMB_AMD] | None = None,
	**kwargs,
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
		TYPE_EMB_ALL: The embedding.
	"""
	try:
		return xtal.get_embedding(distance, **kwargs)
	except Exception as e:
		return e


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
	xtal_1 = Crystal.from_Structure(xtal_1) if isinstance(xtal_1, Structure) else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if isinstance(xtal_2, Structure) else xtal_2

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
	elif distance == "elmd":
		d = _d_elmd(emb_1, emb_2, **(kwargs.get("args_dist", {})))
	else:
		raise ValueError(f"Unsupported distance metric: {distance}")

	# normalize distance if needed
	if distance in CONTINUOUS_UNNORMALIZED_DISTANCES and normalize:
		d = d / (1 + d)

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
			using the embeddings.

	Returns:
		TYPE_D_MTX_RETURN: Distance matrix, the embeddings of xtals_1 (and xtals_2 if
		xtals_2 is not None) and the computing time.

	Raises:
		ValueError: If an unsupported distance metric is provided.

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
	if distance == "smat":
		d_mtx = _distance_matrix_d_smat(
			embs_1,
			embs_2,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_dist", {})),
		)
	elif distance == "comp":
		d_mtx = _distance_matrix_d_comp(embs_1, embs_2, multiprocessing, n_processes)
	elif distance == "wyckoff":
		d_mtx = _distance_matrix_d_wyckoff(embs_1, embs_2, multiprocessing, n_processes)
	elif distance == "magpie":
		d_mtx = _distance_matrix_d_magpie(embs_1, embs_2, multiprocessing, n_processes)
	elif distance == "pdd":
		d_mtx = _distance_matrix_d_pdd(
			embs_1,
			embs_2,
			multiprocessing,
			n_processes,
			**(kwargs.get("args_dist", {})),
		)
	elif distance == "amd":
		if multiprocessing:
			warnings.warn(
				"Multiprocessing is not implemented for _distance_matrix_d_amd. "
				"Proceeding without multiprocessing.",
				stacklevel=2,
			)
		d_mtx = _distance_matrix_d_amd(embs_1, embs_2, **(kwargs.get("args_dist", {})))
	elif distance == "elmd":
		d_mtx = _distance_matrix_d_elmd(
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
		d_mtx = d_mtx / (1 + d_mtx)

	# return results
	if not verbose:
		return d_mtx
	elif given_two_sets:
		return d_mtx, embs_1, embs_2, times
	else:
		return d_mtx, embs_1, times
