"""This module offers a range of distance functions for crystals."""

import warnings

import amd
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .crystal import Crystal


def d_smat(xtal_1: Structure | Crystal, xtal_2: Structure | Crystal, **kwargs) -> float:
	"""Compute the binary distance based on pymatgen's StructureMatcher.

	Args:
		xtal_1 (Structure | Crystal): pymatgen Structure or Crystal.
		xtal_2 (Structure | Crystal): pymatgen Structure or Crystal.
		**kwargs: Additional arguments for StructureMatcher, e.g., ltol, stol,
			angle_tol.

	Returns:
		float: StructureMatcher-based distance (0.0 or 1.0).

	Note:
		d_smat does not allow pre-computation of embeddings.
	"""
	matcher = StructureMatcher(**kwargs)
	return 0.0 if matcher.fit(xtal_1, xtal_2) else 1.0


def d_comp(
	xtal_1: Structure | Crystal | tuple[tuple[str, int]],
	xtal_2: Structure | Crystal | tuple[tuple[str, int]],
) -> float:
	"""Compute the binary distance based on the match of compositions.

	Args:
		xtal_1 (Structure | Crystal | tuple[tuple[str, int]]): pymatgen Structure or
			Crystal or its embedding.
		xtal_2 (Structure | Crystal | tuple[tuple[str, int]]): pymatgen Structure or
			Crystal or its embedding.

	Returns:
		float: Composition distance (0.0 or 1.0).
	"""
	xtal_1 = Crystal.from_Structure(xtal_1) if type(xtal_1) is Structure else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if type(xtal_2) is Structure else xtal_2
	emb_1 = xtal_1.get_composition_tuple() if type(xtal_1) is Crystal else xtal_1
	emb_2 = xtal_2.get_composition_tuple() if type(xtal_2) is Crystal else xtal_2
	return 0.0 if emb_1 == emb_2 else 1.0


def d_wyckoff(
	xtal_1: Structure | Crystal | tuple[int, tuple[str]],
	xtal_2: Structure | Crystal | tuple[int, tuple[str]],
) -> float | Exception:
	"""Compute the binary distance based on the match of Wyckoff representations.

	Args:
		xtal_1 (Structure | Crystal | tuple[int, tuple[str]]): pymatgen Structure or
			Crystal or its embedding.
		xtal_2 (Structure | Crystal | tuple[int, tuple[str]]): pymatgen Structure or
			Crystal or its embedding.

	Returns:
		float: Wyckoff distance (0.0 or 1.0).
	"""
	xtal_1 = Crystal.from_Structure(xtal_1) if type(xtal_1) is Structure else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if type(xtal_2) is Structure else xtal_2
	try:
		emb_1 = xtal_1.get_wyckoff() if type(xtal_1) is Crystal else xtal_1
	except Exception as e:
		print(f"Failed to get Wyckoff representation of xtal_1. Error message: {e}")
		return -1.0
	try:
		emb_2 = xtal_2.get_wyckoff() if type(xtal_2) is Crystal else xtal_2
	except Exception as e:
		print(f"Failed to get Wyckoff representation of xtal_2. Error message: {e}")
		return -1.0
	return 0.0 if emb_1 == emb_2 else 1.0


def d_magpie(
	xtal_1: Structure | Crystal | list[float], xtal_2: Structure | Crystal | list[float]
) -> float:
	"""Compute the continuous distance using compositional Magpie fingerprints.

	Args:
		xtal_1 (Structure | Crystal | list[float]): pymatgen Structure or Crystal or its
	        embedding.
		xtal_2 (Structure | Crystal | list[float]): pymatgen Structure or Crystal or its
			embedding.

	Returns:
		float: Magpie distance.

	References:
		- Ward et al., (2016). A general-purpose machine learning framework for
			predicting properties of inorganic materials. npj Computational Materials,
			2(1), 1-7. https://doi.org/10.1038/npjcompumats.2016.28
	"""
	xtal_1 = Crystal.from_Structure(xtal_1) if type(xtal_1) is Structure else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if type(xtal_2) is Structure else xtal_2
	emb_1 = (
		np.array(xtal_1.get_magpie())
		if isinstance(xtal_1, Crystal)
		else np.array(xtal_1)
	)
	emb_2 = (
		np.array(xtal_2.get_magpie())
		if isinstance(xtal_2, Crystal)
		else np.array(xtal_2)
	)
	return np.sqrt(np.sum((emb_1 - emb_2) ** 2)).item()


def d_pdd(
	xtal_1: Structure | Crystal | np.ndarray[np.float32 | np.float64],
	xtal_2: Structure | Crystal | np.ndarray[np.float32 | np.float64],
	**kwargs,
) -> float | Exception:
	"""Compute the continuous distance using the pointwise distance distribution (PDD).

	Args:
		xtal_1 (Structure | Crystal | np.ndarray[np.float32 | np.float64]): pymatgen
			Structure or Crystal or its embedding.
		xtal_2 (Structure | Crystal | np.ndarray[np.float32 | np.float64]): pymatgen
			Structure or Crystal or its embedding.
		**kwargs: Additional arguments for amd.PDD and amd.PDD_cdist. It can contain
			two keys: "emb" and "dist". The value of "emb" is a dict of arguments for
			amd.PDD, and the value of "dist" is a dict of arguments for amd.PDD_cdist.

	Returns:
		float: PDD distance.

	Examples:
		>>> kwargs = {
		...     "emb": {"k": 100},
		...     "dist": {
		...         "metric": "chebyshev",
		...         "backend": "multiprocessing",
		...         "n_jobs": 2,
		...         "verbose": False,
		...     },
		... }
		>>> d_pdd(xtal_1, xtal_2, **kwargs)
		0.123456789

	References:
		- Widdowson et al., (2022). Resolving the data ambiguity for periodic
			crystals. Advances in Neural Information Processing Systems, 35,
			24625--24638. https://openreview.net/forum?id=4wrB7Mo9_OQ
	"""
	xtal_1 = Crystal.from_Structure(xtal_1) if type(xtal_1) is Structure else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if type(xtal_2) is Structure else xtal_2
	if "emb" not in kwargs:
		kwargs["emb"] = {}
	if "dist" not in kwargs:
		kwargs["dist"] = {}
	try:
		emb_1 = xtal_1.get_PDD(**kwargs["emb"]) if type(xtal_1) is Crystal else xtal_1
	except Exception as e:
		warnings.warn(
			f"Failed to get the PDD representation of xtal_1. Error message: {e}",
			UserWarning,
			stacklevel=2,
		)
		return -1.0
	try:
		emb_2 = xtal_2.get_PDD(**kwargs["emb"]) if type(xtal_2) is Crystal else xtal_2
	except Exception as e:
		warnings.warn(
			f"Failed to get the PDD representation of xtal_2. Error message: {e}",
			UserWarning,
			stacklevel=2,
		)
		return -1.0
	return amd.PDD_cdist([emb_1], [emb_2], **kwargs["dist"])[0][0].item()


def d_amd(
	xtal_1: Structure | Crystal | np.ndarray[np.float32 | np.float64],
	xtal_2: Structure | Crystal | np.ndarray[np.float32 | np.float64],
	**kwargs,
) -> float:
	"""Compute the continuous distance using the average minimum distance (AMD).

	Args:
		xtal_1 (Structure | Crystal | np.ndarray[np.float32 | np.float64]): pymatgen
			Structure or Crystal or its embedding.
		xtal_2 (Structure | Crystal | np.ndarray[np.float32 | np.float64]): pymatgen
			Structure or Crystal or its embedding.
		**kwargs: Additional arguments for amd.AMD and amd.AMD_cdist. It should contain
			two keys: "emb" and "dist". The value of "emb" is a dict of arguments for
			amd.AMD, and the value of "dist" is a dict of arguments for amd.AMD_cdist.

	Returns:
		float: AMD distance.

	Examples:
		>>> kwargs = {
		...     "emb": {"k": 100},
		...     "dist": {"metric": "chebyshev", "low_memory": False},
		... }
		>>> d_amd(xtal_1, xtal_2, **kwargs)
		0.123456789

	References:
		- Widdson et al., (2022). Average Minimum Distances of periodic point sets -
			foundational invariants for mapping periodic crystals. MATCH
			Communications in Mathematical and in Computer Chemistry, 87(3), 529-559,
			https://doi.org/10.46793/match.87-3.529W
	"""
	xtal_1 = Crystal.from_Structure(xtal_1) if type(xtal_1) is Structure else xtal_1
	xtal_2 = Crystal.from_Structure(xtal_2) if type(xtal_2) is Structure else xtal_2
	if "emb" not in kwargs:
		kwargs["emb"] = {}
	if "dist" not in kwargs:
		kwargs["dist"] = {}
	try:
		emb_1 = xtal_1.get_AMD(**kwargs["emb"]) if type(xtal_1) is Crystal else xtal_1
	except Exception as e:
		warnings.warn(
			f"Failed to get the AMD representation of xtal_1. Error message: {e}",
			UserWarning,
			stacklevel=2,
		)
		return -1.0
	try:
		emb_2 = xtal_2.get_AMD(**kwargs["emb"]) if type(xtal_2) is Crystal else xtal_2
	except Exception as e:
		warnings.warn(
			f"Failed to get the AMD representation of xtal_2. Error message: {e}",
			UserWarning,
			stacklevel=2,
		)
		return -1.0
	return amd.AMD_cdist([emb_1], [emb_2], **kwargs["dist"])[0][0].item()
