"""This module contains the Evaluator class for uniqueness and novelty calculation."""

import gzip
import os
import pickle
import time
from typing import Any, Literal

import numpy as np
from huggingface_hub import hf_hub_download
from pymatgen.core import Structure

from .constants import (
	BINARY_DISTANCES,
	CONTINUOUS_DISTANCES,
	HF_VERSION,
	SUPPORTED_DISTANCES,
	SUPPORTED_SCREENS,
)
from .crystal import Crystal
from .distance import distance_matrix
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
		self.n_samples = len(self.gen_xtals)

	def _read_pickle_gz(self, path: str) -> Any:
		"""Load data from a pkl.gz file.

		Args:
			path (str): Path to the pkl.gz file.

		Returns:
			Any: Loaded data.
		"""
		with gzip.open(path, "rb") as f:
			data = pickle.load(f)
		return data

	def _write_pickle_gz(self, data: Any, path: str) -> None:
		"""Save data to a pkl.gz file.

		Args:
			data (Any): Data to be saved.
			path (str): Path to the pkl.gz file.
		"""
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with gzip.open(path, "wb") as f:
			pickle.dump(data, f)

	def uniqueness(
		self,
		distance: str,
		screen: str | None = None,
		dir_intermediate_gen: str | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		"""Evaluate the uniqueness of a set of crystals.

		Args:
			distance (str): Distance function used for uniqueness evaluation. Currently
				supported distances are shown in SUPPORTED_DISTANCES in constants.py.
				For more detailed information about each distance metric, please refer
				to the `tutorial notebook`_.
			screen (str | None): Method to screen the crystals. Currently supported
				methods are shown in SUPPORTED_SCREENS in constants.py.
			dir_intermediate_gen (str | None): Directory to search for pre-computed
				embeddings, distance matrix, and screening results for faster
				computation. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved. It is recommended that you set this argument.
				This is especially important when evaluating a large number of generated
				crystals or when d_smat is used as the distance metric.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				thermodynamic screening. It can contain three keys: "args_emb",
				"args_dist", and "args_screen". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_dist" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_screen" is a dict of arguments for
				the screening function.

		Examples:
			>>> evaluator.uniqueness(
			...     distance="smat",
			...     screen=None,
			...     dir_intermediate_gen="./intermediate",
			...     return_time=True,
			... )
			>>> (
			...     0.9945,
			...     {
			...         "uni_emb": 0.0,
			...         "uni_d_mtx": 16953.978,
			...         "uni_metric": 0.152,
			...         "uni_total": 16954.130,
			...     },
			... )
			>>> evaluator.uniqueness(
			...     distance="amd",
			...     screen="ehull",
			...     dir_intermediate_gen="./intermediate",
			...     return_time=False,
			...     **{
			...         "args_emb": {"k": 200},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...         "args_screen": {"diagram": "mp_250618"},
			...     },
			... )
			>>> 0.0016

		Returns:
			float | tuple: Uniqueness value or (uniqueness value, a dictionary of time
			taken for each step).

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		if distance not in SUPPORTED_DISTANCES:
			raise ValueError(f"Unsupported distance: {distance}.")
		if screen not in SUPPORTED_SCREENS:
			raise ValueError(f"Unsupported screening method: {screen}.")

		times: dict[str, float] = {}

		# Step 1: Compute distance matrix
		# Use pre-computed distance matrix if available
		if dir_intermediate_gen is not None and os.path.exists(
			os.path.join(dir_intermediate_gen, f"mtx_uni_{distance}.pkl.gz")
		):
			d_mtx = self._read_pickle_gz(
				os.path.join(dir_intermediate_gen, f"mtx_uni_{distance}.pkl.gz")
			)
			times["uni_emb"] = 0.0
			times["uni_d_mtx"] = 0.0
		else:
			# Prepare generated samples
			gen_embed = False
			if dir_intermediate_gen is not None and os.path.exists(
				os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz")
			):
				gen_xtals = self._read_pickle_gz(
					os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz")
				)
				gen_embed = True
			else:
				gen_xtals = self.gen_xtals
			# Distance matrix computation
			d_mtx, embs_gen, times_matrix = distance_matrix(
				distance,
				gen_xtals,
				None,
				True,
				**kwargs,
			)
			# Record times
			times["uni_emb"] = times_matrix["emb_1"]
			times["uni_d_mtx"] = times_matrix["d_mtx"]
			# Save intermediate results
			if dir_intermediate_gen is not None:
				self._write_pickle_gz(
					d_mtx,
					os.path.join(dir_intermediate_gen, f"mtx_uni_{distance}.pkl.gz"),
				)
				if not gen_embed:
					self._write_pickle_gz(
						embs_gen,
						os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz"),
					)

		# Step 2: Screening (optional)
		valid_indices = np.ones(self.n_samples, dtype=bool)
		# Remove crystals whose embeddings could not be computed
		valid_indices &= np.array(
			[d_mtx_i0 != float("nan") for d_mtx_i0 in d_mtx[:, 0]]
		)
		if screen == "smact":
			valid_indices &= screen_smact(self.gen_xtals, dir_intermediate_gen)
		elif screen == "ehull":
			valid_indices &= screen_ehull(
				self.gen_xtals,
				diagram=kwargs.get("args_screen", {"diagram": "mp_250618"})["diagram"],
				dir_intermediate=dir_intermediate_gen,
			)
		d_mtx = d_mtx[valid_indices][:, valid_indices]

		# Step 3: Compute uniqueness
		start_time_metric = time.time()
		if distance in BINARY_DISTANCES:
			n_unique = sum(
				[1 if np.all(d_mtx[i, :i] != 0) else 0 for i in range(len(d_mtx))]
			)
			uniqueness = n_unique / self.n_samples
		elif distance in CONTINUOUS_DISTANCES:
			uniqueness = float(np.sum(d_mtx) / (self.n_samples * (self.n_samples - 1)))
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
		distance: str,
		screen: str | None = None,
		dir_intermediate_gen: str | None = None,
		dir_intermediate_train: str | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		"""Evaluate the novelty of a set of crystals.

		Args:
			train_xtals (list[Crystal | Structure] | Literal["mp20"]): List of training
				crystal structures or dataset name. If a dataset name is given, the
				embeddings of its training data will be downloaded from Hugging Face.
				The embeddings were computed using the _compute_embeddings function in
				distance.py with no additional kwargs.
			distance (str): Distance used for novelty evaluation. Currently supported
				distances are shown in SUPPORTED_DISTANCES in constants.py. For more
				detailed information about each distance metric, please refer to the
				`tutorial notebook`_.
			screen (str | None): Method to screen the generated crystals. Currently
				supported methods are shown in SUPPORTED_SCREENS in constants.py.
			dir_intermediate_gen (str | None): Directory to search for pre-computed
				embeddings, distance matrix, and screening results for faster
				computation. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved. It is recommended that you set this argument.
				This is especially important when evaluating a large number of generated
				crystals or when d_smat is used as the distance metric.
			dir_intermediate_train (str | None): Directory to search for pre-computed
				embeddings of training crystals. If pre-computed files do not exist in
				the directory, they will be saved to the directory for future use. If
				set to None, no files will be loaded or saved. It is recommended that
				you set this argument. This is especially important when evaluating a
				large number of generated crystals. If train_xtals is a dataset name,
				this argument is ignored.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				thermodynamic screening. It can contain three keys: "args_emb",
				"args_dist", and "args_screen". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_dist" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_screen" is a dict of arguments for
				the screening function.

		Examples:
			>>> evaluator.novelty(
			...     train_xtals="mp20",
			...     distance="smat",
			...     screen=None,
			...     dir_intermediate_gen="./intermediate",
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
			...     dir_intermediate_gen="./intermediate",
			...     dir_intermediate_train="./intermediate_train",
			...     return_time=False,
			...     **{
			...         "args_emb": {"k": 200},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...         "args_screen": {"diagram": "mp_250618"},
			...     },
			... )
			>>> 0.0075

		Returns:
			float | tuple: Novelty value or a tuple containing the novelty value
				and a dictionary of time taken for each step.

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		if isinstance(train_xtals, str) and train_xtals not in ["mp20"]:
			raise ValueError(f"Unsupported dataset name: {train_xtals}.")
		if distance not in SUPPORTED_DISTANCES:
			raise ValueError(f"Unsupported distance: {distance}.")
		if screen not in SUPPORTED_SCREENS:
			raise ValueError(f"Unsupported screening method: {screen}.")

		times: dict[str, float] = {}

		# Step 1: Compute distance matrix
		# Use pre-computed distance matrix if available
		if dir_intermediate_gen is not None and os.path.exists(
			os.path.join(dir_intermediate_gen, f"mtx_nov_{distance}.pkl.gz")
		):
			d_mtx = self._read_pickle_gz(
				os.path.join(dir_intermediate_gen, f"mtx_nov_{distance}.pkl.gz")
			)
			times["nov_emb_gen"] = 0.0
			times["nov_emb_train"] = 0.0
			times["nov_d_mtx"] = 0.0
		else:
			# Prepare generated samples
			gen_embed = False
			if dir_intermediate_gen is not None and os.path.exists(
				os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz")
			):
				gen_xtals = self._read_pickle_gz(
					os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz")
				)
				gen_embed = True
			else:
				gen_xtals = self.gen_xtals
			# Prepare training samples
			train_embed = False
			if not isinstance(train_xtals, list):
				path_embs_train = hf_hub_download(
					repo_id="masahiro-negishi/xtalmet",
					filename=f"mp20/train/train_{distance}.pkl.gz",
					repo_type="dataset",
					revision=HF_VERSION,
				)
				train_xtals = self._read_pickle_gz(path_embs_train)
				times["nov_emb_train"] = 0.0
				train_embed = True
			elif dir_intermediate_train is not None and os.path.exists(
				os.path.join(dir_intermediate_train, f"train_{distance}.pkl.gz")
			):
				train_xtals = self._read_pickle_gz(
					os.path.join(dir_intermediate_train, f"train_{distance}.pkl.gz")
				)
				train_embed = True
			else:
				train_xtals = [
					xtal if isinstance(xtal, Crystal) else Crystal.from_Structure(xtal)
					for xtal in train_xtals
				]
			# Distance matrix computation
			d_mtx, embs_gen, embs_train, times_matrix = distance_matrix(
				distance,
				gen_xtals,
				train_xtals,
				True,
				**kwargs,
			)
			# Record times
			times["nov_emb_gen"] = times_matrix["emb_1"]
			times["nov_emb_train"] = times_matrix["emb_2"]
			times["nov_d_mtx"] = times_matrix["d_mtx"]
			# Save intermediate results
			if dir_intermediate_gen is not None:
				self._write_pickle_gz(
					d_mtx,
					os.path.join(dir_intermediate_gen, f"mtx_nov_{distance}.pkl.gz"),
				)
				if not gen_embed:
					self._write_pickle_gz(
						embs_gen,
						os.path.join(dir_intermediate_gen, f"gen_{distance}.pkl.gz"),
					)
			if (
				isinstance(train_xtals, list)
				and dir_intermediate_train is not None
				and not train_embed
			):
				self._write_pickle_gz(
					embs_train,
					os.path.join(dir_intermediate_train, f"train_{distance}.pkl.gz"),
				)

		# Step 2: Screening (optional)
		valid_indices_gen = np.ones(self.n_samples, dtype=bool)
		valid_indices_train = np.ones(len(d_mtx[0]), dtype=bool)
		# Remove crystals whose embeddings could not be computed
		valid_indices_gen &= np.array(
			[d_mtx_i0 != float("nan") for d_mtx_i0 in d_mtx[:, 0]]
		)
		valid_indices_train &= np.array(
			[d_mtx_0j != float("nan") for d_mtx_0j in d_mtx[0]]
		)
		if screen == "smact":
			valid_indices_gen &= screen_smact(self.gen_xtals, dir_intermediate_gen)
		elif screen == "ehull":
			valid_indices_gen &= screen_ehull(
				self.gen_xtals,
				diagram=kwargs.get("args_screen", {"diagram": "mp_250618"})["diagram"],
				dir_intermediate=dir_intermediate_gen,
			)
		d_mtx = d_mtx[valid_indices_gen][:, valid_indices_train]

		# Step 3: Compute novelty
		start_time_metric = time.time()
		if distance in BINARY_DISTANCES:
			n_novel = sum(
				[1 if np.all(d_mtx[i] != 0) else 0 for i in range(len(d_mtx))]
			)
			novelty = n_novel / self.n_samples
		elif distance in CONTINUOUS_DISTANCES:
			novelty = float(np.sum(np.min(d_mtx, axis=1)) / self.n_samples)
		end_time_metric = time.time()
		times["nov_metric"] = end_time_metric - start_time_metric
		times["nov_total"] = sum(times.values())

		if return_time:
			return novelty, times
		else:
			return novelty
