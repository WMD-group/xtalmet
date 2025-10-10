"""This module contains the Evaluator class for uniqueness and novelty calculation."""

import gzip
import os
import pickle
import time
from typing import Any, Literal

import amd
import numpy as np
from huggingface_hub import hf_hub_download
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.spatial.distance import squareform

from .crystal import Crystal
from .screen import screen_ehull, screen_smact


class Evaluator:
	"""Class for storing and evaluating a set of crystals."""

	def __init__(
		self,
		gen_xtals: list[Crystal | Structure],
	) -> None:
		"""Initialize the Evaluator.

		Args:
			gen_xtals (list[Crystal | Structure]): Generated crystal structures.
		"""
		assert all(isinstance(xtal, (Crystal, Structure)) for xtal in gen_xtals), (
			"All elements in gen_xtals must be of type Crystal or pymatgen Structure."
		)
		self.gen_xtals = [
			xtal if isinstance(xtal, Crystal) else Crystal.from_Structure(xtal)
			for xtal in gen_xtals
		]

	def _embed(
		self,
		xtals: list[Crystal],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		dir_intermediate: str | None = None,
		datatype: Literal["gen", "train"] | None = None,
		**kwargs,
	) -> tuple[list[Any], float]:
		"""Compute the embeddings of crystals.

		Args:
			xtals (list[Crystal]): List of crystal structures to embed.
			distance (Literal): Type of distance.
			dir_intermediate (str | None): Directory to search for pre-computed
				embeddings. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved.
			datatype (Literal["gen", "train"] | None): If you specify dir_intermediate,
				you must also specify whether xtals are generated crystals or training
				crystals. This helps to avoid file name conflicts. If dir_intermediate
				is None, this argument is ignored.
			**kwargs: Additional keyword arguments for d_pdd and d_amd.

		Returns:
			tuple: A tuple containing the embeddings of crystals
				   and the time taken to compute them.
		"""
		if dir_intermediate is not None:
			path_intermediate = os.path.join(
				dir_intermediate, f"{datatype}_{distance}.pkl.gz"
			)

		start_time_emb = time.time()
		if dir_intermediate is not None and os.path.exists(path_intermediate):
			with gzip.open(path_intermediate, "rb") as f:
				embs = pickle.load(f)
		else:
			if distance == "smat":
				embs = [xtal for xtal in xtals]
			if distance == "comp":
				embs = [xtal.get_composition_tuple() for xtal in xtals]
			elif distance == "wyckoff":
				embs = []
				for xtal in xtals:
					try:
						embs.append(xtal.get_wyckoff())
					except Exception as e:
						embs.append(e)
			elif distance == "magpie":
				embs = [xtal.get_magpie() for xtal in xtals]
			elif distance == "pdd":
				embs = []
				for xtal in xtals:
					try:
						embs.append(xtal.get_PDD(**kwargs))
					except Exception as e:
						embs.append(e)
			elif distance == "amd":
				embs = []
				for xtal in xtals:
					try:
						embs.append(xtal.get_AMD(**kwargs))
					except Exception as e:
						embs.append(e)
		end_time_emb = time.time()

		if dir_intermediate is not None and not os.path.exists(path_intermediate):
			os.makedirs(dir_intermediate, exist_ok=True)
			with gzip.open(path_intermediate, "wb") as f:
				pickle.dump(embs, f)

		time_emb = end_time_emb - start_time_emb
		return embs, time_emb

	def _distance_matrix(
		self,
		embs_1: list[Any],
		embs_2: list[Any],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		dir_intermediate: str | None = None,
		met: Literal["uni", "nov"] | None = None,
		**kwargs,
	) -> tuple[np.ndarray, float]:
		"""Compute the distance matrix for the embeddings.

		Args:
			embs_1 (list[Any]): List of embeddings for the first set of crystals.
			embs_2 (list[Any]): List of embeddings for the second set of crystals.
			distance (Literal): Type of distance.
			dir_intermediate (str | None): Directory to search for pre-computed
				distance matrix. If pre-computed files do not exist in the directory,
				they will be saved to the directory for future use. If set to None, no
				files will be loaded or saved.
			met (Literal["uni", "nov"] | None): If you specify dir_intermediate,
				you must also specify whether the distance matrix is for uniqueness or
				novelty evaluation. This helps to avoid file name conflicts. If
				dir_intermediate is None, this argument is ignored.
			**kwargs: Additional keyword arguments for d_smat, d_pdd, and d_amd.

		Returns:
			tuple[np.ndarray, float]: A tuple containing the distance matrix
				and the time taken to compute it.
		"""
		if dir_intermediate is not None:
			path_intermediate = os.path.join(
				dir_intermediate, f"mtx_{met}_{distance}.pkl.gz"
			)

		start_time_matrix = time.time()
		if dir_intermediate is not None and os.path.exists(path_intermediate):
			with gzip.open(path_intermediate, "rb") as f:
				d_mtx = pickle.load(f)
		else:
			if distance == "smat":
				d_mtx = np.ones((len(embs_1), len(embs_2)))
				matcher = StructureMatcher(**kwargs)
				for i, emb_i in enumerate(embs_1):
					for j, emb_j in enumerate(embs_2):
						if matcher.fit(emb_i, emb_j):
							d_mtx[i, j] = 0
			elif distance == "comp":
				d_mtx = np.ones((len(embs_1), len(embs_2)))
				for i, emb_i in enumerate(embs_1):
					for j, emb_j in enumerate(embs_2):
						if emb_i == emb_j:
							d_mtx[i, j] = 0
			elif distance == "wyckoff":
				d_mtx = np.ones((len(embs_1), len(embs_2)))
				for i, emb_i in enumerate(embs_1):
					if isinstance(emb_i, Exception):  # error
						d_mtx[i, :] = -1
					for j, emb_j in enumerate(embs_2):
						if isinstance(emb_j, Exception):  # error
							d_mtx[i, j] = -1
						elif emb_i == emb_j:
							d_mtx[i, j] = 0
			elif distance == "magpie":
				d_mtx = np.zeros((len(embs_1), len(embs_2)))
				embs_1 = np.array(embs_1)
				embs_2 = np.array(embs_2)
				for i, emb in enumerate(embs_1):
					d_sq = (emb[np.newaxis, :] - embs_2) ** 2
					d_euclidean = np.sqrt(np.sum(d_sq, axis=1))
					d_mtx[i, :] = d_euclidean
			elif distance in ["pdd", "amd"]:
				valids_1 = [x for x in embs_1 if isinstance(x, np.ndarray)]
				error_indices_1 = [
					i for i, x in enumerate(embs_1) if isinstance(x, Exception)
				]
				valids_2 = [x for x in embs_2 if isinstance(x, np.ndarray)]
				error_indices_2 = [
					i for i, x in enumerate(embs_2) if isinstance(x, Exception)
				]
				if met == "uni":
					d_mtx = squareform(
						amd.PDD_pdist(valids_1, **kwargs)
						if distance == "pdd"
						else amd.AMD_pdist(valids_1, **kwargs)
					)
				else:
					d_mtx = (
						amd.PDD_cdist(valids_1, valids_2, **kwargs)
						if distance == "pdd"
						else amd.AMD_cdist(valids_1, valids_2, **kwargs)
					)
				for i in error_indices_1:
					d_mtx = np.insert(d_mtx, i, -1, axis=0)
				for i in error_indices_2:
					d_mtx = np.insert(d_mtx, i, -1, axis=1)
				assert d_mtx.shape == (len(embs_1), len(embs_2))
		end_time_matrix = time.time()

		if dir_intermediate is not None and not os.path.exists(path_intermediate):
			os.makedirs(dir_intermediate, exist_ok=True)
			with gzip.open(path_intermediate, "wb") as f:
				pickle.dump(d_mtx, f)

		time_matrix = end_time_matrix - start_time_matrix
		return d_mtx, time_matrix

	def uniqueness(
		self,
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		screen: Literal[None, "smact", "ehull"] = None,
		dir_intermediate: str | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		"""Evaluate the uniqueness of a set of crystals.

		Args:
			distance (Literal): Distance function used for uniqueness evaluation.
			screen (Literal): Method to screen the crystals.
			dir_intermediate (str | None): Directory to search for pre-computed
				embeddings, distance matrix, and screening results for faster
				computation. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved. It is recommended that you set this argument.
				This is especially important when evaluating a large number of generated
				crystals or when d_smat is used as the distance metric.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				thermodynamic screening. It can contain three keys: "args_emb",
				"args_mtx", and "args_screen". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_mtx" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_screen" is a dict of arguments for
				the screening function.

		Examples:
			>>> evaluator.uniqueness(
			...     distance="smat",
			...     screen=None,
			...     dir_intermediate="./intermediate",
			...     return_time=True,
			... )
			>>> (
			...     0.9945,
			...     {
			...         "uni_emb": 0.003,
			...         "uni_d_mtx": 16953.978,
			...         "uni_metric": 0.152,
			...         "uni_total": 16954.133,
			...     },
			... )
			>>> evaluator.uniqueness(
			...     distance="amd",
			...     screen="ehull",
			...     dir_intermediate="./intermediate",
			...     return_time=False,
			...     kwargs={
			...         "args_emb": {"k": 200},
			...         "args_mtx": {"metric": "chebyshev", "low_memory": False},
			...         "args_screen": {"diagram": "mp_250618"},
			...     },
			... )
			>>> 0.0016

		Returns:
			float | tuple: Uniqueness value or (uniqueness value, a dictionary of time
			taken for each step).
		"""
		if distance not in ["smat", "comp", "wyckoff", "magpie", "pdd", "amd"]:
			raise ValueError(f"Unsupported distance: {distance}.")
		if screen not in [None, "smact", "ehull"]:
			raise ValueError(f"Unsupported screening method: {screen}.")

		times: dict[str, float] = {}

		# Step 1: Compute embeddings
		embs, time_emb = self._embed(
			self.gen_xtals,
			distance,
			dir_intermediate,
			"gen",
			**(kwargs.get("args_emb", {})),
		)
		times["uni_emb"] = time_emb

		# Step 2: Compute distance matrix
		d_mtx, time_matrix = self._distance_matrix(
			embs,
			embs,
			distance,
			dir_intermediate,
			"uni",
			**(kwargs.get("args_mtx", {})),
		)
		times["uni_d_mtx"] = time_matrix

		# Step 3: Screening (optional)
		valid_indices = np.ones(len(embs), dtype=bool)
		if distance in ["wyckoff", "pdd", "amd"]:
			# Remove crystals whose embeddings could not be computed
			valid_indices &= np.array([not isinstance(x, Exception) for x in embs])
		if screen == "smact":
			valid_indices &= screen_smact(self.gen_xtals, dir_intermediate)
		elif screen == "ehull":
			valid_indices &= screen_ehull(
				self.gen_xtals,
				diagram=kwargs.get("args_screen", {"diagram": "mp_250618"})["diagram"],
				dir_intermediate=dir_intermediate,
			)
		d_mtx = d_mtx[valid_indices][:, valid_indices]

		# Step 4: Compute uniqueness
		start_time_metric = time.time()
		if distance in ["smat", "comp", "wyckoff"]:
			n_unique = sum(
				[1 if np.all(d_mtx[i, :i] != 0) else 0 for i in range(len(d_mtx))]
			)
			uniqueness = n_unique / len(embs)
		elif distance in ["magpie", "pdd", "amd"]:
			uniqueness = float(np.sum(d_mtx) / (len(embs) * (len(embs) - 1)))
		end_time_metric = time.time()
		times["uni_metric"] = end_time_metric - start_time_metric
		times["uni_total"] = sum(times.values())

		if return_time:
			return uniqueness, times
		else:
			return uniqueness

	def novelty(
		self,
		train_xtals: list[Crystal | Structure] | Literal["mp20"],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		screen: Literal[None, "smact", "ehull"] = None,
		dir_intermediate: str | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		"""Evaluate the novelty of a set of crystals.

		Args:
			train_xtals (list[Crystal | Structure] | Literal["mp20"]): List of training
				crystal structures or dataset name. If a dataset name is given, the
				embeddings of its training data will be downloaded from Hugging Face.
				The embeddings were computed using the _embed method above with no
				additional kwargs.
			distance (Literal): Distance used for novelty evaluation.
			screen (Literal): Method to screen the generated crystals.
			dir_intermediate (str | None): Directory to search for pre-computed
				embeddings, distance matrix, and screening results for faster
				computation. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved. It is recommended that you set this argument.
				This is especially important when evaluating a large number of generated
				crystals or when d_smat is used as the distance metric.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				thermodynamic screening. It can contain three keys: "args_emb",
				"args_mtx", and "args_screen". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_mtx" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_screen" is a dict of arguments for
				the screening function.

		Examples:
			>>> evaluator.novelty(
			...     train_xtals="mp20",
			...     distance="smat",
			...     screen=None,
			...     dir_intermediate="./intermediate",
			...     return_time=True,
			... )
			>>> (
			...     0.9892,
			...     {
			...         "nov_emb_gen": 1.693,
			...         "nov_emb_train": 5.790,
			...         "nov_d_mtx": 42784.921,
			...         "nov_metric": 0.628,
			...         "nov_total": 42793.032,
			...     },
			... )
			>>> evaluator.novelty(
			...     train_xtals=list_of_train_xtals,
			...     distance="amd",
			...     screen="ehull",
			...     dir_intermediate="./intermediate",
			...     return_time=False,
			...     kwargs={
			...         "args_emb": {"k": 200},
			...         "args_mtx": {"metric": "chebyshev", "low_memory": False},
			...         "args_screen": {"diagram": "mp_250618"},
			...     },
			... )
			>>> 0.0075

		Returns:
			float | tuple: Novelty value or a tuple containing the novelty value
				and a dictionary of time taken for each step.
		"""
		if isinstance(train_xtals, str) and train_xtals not in ["mp20"]:
			raise ValueError(f"Unsupported dataset name: {train_xtals}.")
		if distance not in ["smat", "comp", "wyckoff", "magpie", "pdd", "amd"]:
			raise ValueError(f"Unsupported distance: {distance}.")
		if screen not in [None, "smact", "ehull"]:
			raise ValueError(f"Unsupported screening method: {screen}.")

		times: dict[str, float] = {}

		# Step 1: Compute embeddings
		embs_gen, time_emb_gen = self._embed(
			self.gen_xtals,
			distance,
			dir_intermediate,
			"gen",
			**(kwargs.get("args_emb", {})),
		)
		times["nov_emb_gen"] = time_emb_gen
		if isinstance(train_xtals, list):
			train_xtals = [
				xtal if isinstance(xtal, Crystal) else Crystal.from_Structure(xtal)
				for xtal in train_xtals
			]
			embs_train, time_emb_train = self._embed(
				train_xtals,
				distance,
				dir_intermediate,
				"train",
				**(kwargs.get("args_emb", {})),
			)
			times["nov_emb_train"] = time_emb_train
		else:
			start_time_emb = time.time()
			path = hf_hub_download(
				repo_id="masahiro-negishi/xtalmet",
				filename=f"mp20/train/train_{distance}.pkl.gz",
				repo_type="dataset",
			)
			with gzip.open(path, "rb") as f:
				embs_train = pickle.load(f)
			end_time_emb = time.time()
			times["nov_emb_train"] = end_time_emb - start_time_emb

		# Step 2: Compute distance matrix
		d_mtx, time_matrix = self._distance_matrix(
			embs_gen,
			embs_train,
			distance,
			dir_intermediate,
			"nov",
			**(kwargs.get("args_mtx", {})),
		)
		times["nov_d_mtx"] = time_matrix

		# Step 3: Screening (optional)
		valid_indices_gen = np.ones(len(embs_gen), dtype=bool)
		valid_indices_train = np.ones(len(embs_train), dtype=bool)
		if distance in ["wyckoff", "pdd", "amd"]:
			# Remove crystals whose embeddings could not be computed
			valid_indices_gen &= np.array(
				[not isinstance(x, Exception) for x in embs_gen]
			)
			valid_indices_train &= np.array(
				[not isinstance(x, Exception) for x in embs_train]
			)
		if screen == "smact":
			valid_indices_gen &= screen_smact(self.gen_xtals, dir_intermediate)
		elif screen == "ehull":
			valid_indices_gen &= screen_ehull(
				self.gen_xtals,
				diagram=kwargs.get("args_screen", {"diagram": "mp_250618"})["diagram"],
				dir_intermediate=dir_intermediate,
			)
		d_mtx = d_mtx[valid_indices_gen][:, valid_indices_train]

		# Step 4: Compute novelty
		start_time_metric = time.time()
		if distance in ["comp", "wyckoff", "smat"]:
			n_novel = sum(
				[1 if np.all(d_mtx[i] != 0) else 0 for i in range(len(d_mtx))]
			)
			novelty = n_novel / len(embs_gen)
		elif distance in ["magpie", "pdd", "amd"]:
			novelty = float(np.sum(np.min(d_mtx, axis=1)) / len(embs_gen))
		end_time_metric = time.time()
		times["nov_metric"] = end_time_metric - start_time_metric
		times["nov_total"] = sum(times.values())

		if return_time:
			return novelty, times
		else:
			return novelty
