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
	SUPPORTED_VALIDITY,
)
from .crystal import Crystal
from .distance import distance_matrix
from .stability import compute_stability_scores
from .validity import screen_smact


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
		validity: list[str] | None = None,
		stability: str | None = None,
		dir_intermediate_gen: str | None = None,
		multiprocessing: bool = False,
		n_processes: int | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		r"""Evaluate the uniqueness of a set of crystals.

		Args:
			distance (str): Distance function used for uniqueness evaluation. Currently
				supported distances are shown in SUPPORTED_DISTANCES in constants.py.
				For more detailed information about each distance metric, please refer
				to the `tutorial notebook`_.
			validity (list[str] | None): Methods to screen the crystals. Currently
				supported methods are shown in SUPPORTED_VALIDITY in constants.py.
			stability (str | None): Stability criterion for the crystals. "continuous"
				or "binary" or None.
			dir_intermediate_gen (str | None): Directory to search for pre-computed
				embeddings, distance matrix, and screening results for faster
				computation. If pre-computed files do not exist in the directory, they
				will be saved to the directory for future use. If set to None, no files
				will be loaded or saved. It is recommended that you set this argument.
				This is especially important when evaluating a large number of generated
				crystals or when d_smat is used as the distance metric.
			multiprocessing (bool): Whether to use multiprocessing for distance matrix
				calculation. Default is False.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing. If None, the number of logical CPU cores - 1 will be
				used. We recommend setting this argument to a smaller number than the
				number of available CPU cores to avoid memory issues. If multiprocessing
				is False, this argument is ignored. Default is None.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				stability calculations. It can contain three keys: "args_emb",
				"args_dist", and "args_stability". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_dist" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_stability" is a dict of arguments for
				the stability evaluation. For more details, please refer to the
				`tutorial notebook`_.

		Examples:
			>>> evaluator.uniqueness(
			...     distance="smat",
			...     validity=None,
			...     stability=None,
			...     dir_intermediate_gen="./intermediate",
			...     multiprocessing=True,
			...     n_processes=10,
			...     return_time=True,
			... )
			>>> (
			...     TODO,
			...     {
			...         "uni_emb": TODO,
			...         "uni_d_mtx": TODO,
			...         "uni_metric": TODO,
			...         "uni_total": TODO,
			...     },
			... )
			>>> evaluator.uniqueness(
			...     distance="pdd",
			...     validity=["smact"],
			...     stability="binary",
			...     dir_intermediate_gen="./intermediate",
			...     multiprocessing=True,
			...     n_processes=10,
			...     return_time=False,
			...     **{
			...         "args_stability": {"diagram": "mp_250618", "threshold": 0.1},
			...     },
			... )
			>>> TODO
			>>> evaluator.uniqueness(
			...     distance="amd",
			...     validity=["smact"],
			...     stability="continuous",
			...     dir_intermediate_gen="./intermediate",
			...     multiprocessing=False,
			...     return_time=False,
			...     **{
			...         "args_emb": {"k": 200},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...         "args_stability": {"diagram": "mp_250618", "intercept": 0.2},
			...     },
			... )
			>>> TODO

		Returns:
			float | tuple: Uniqueness value or (uniqueness value, a dictionary of time
			taken for each step).

		Raises:
			ValueError: If an unsupported distance metric or validity method or
				stability criterion is provided.

		Note:
			Here, I demonstrate how uniqueness is computed for binary/continuous
			distances with binary/continuous stability. The binary stability score,
			:math:`S_b(x)`, for each crystal :math:`x` is defined as follows.

			.. math::
				S_b(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le \text{threshold} \\
						0 & \text{otherwise}
					\end{cases}

			You can set the threshold in kwargs (see the examples). Default threshold is
			0.1 [eV/atom]. The continuous stability score, :math:`S_c(x)`, for each
			crystal :math:`x` is defined as follows.

			.. math::
				S_c(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le 0 \\
						1 - \frac{E_\text{hull}(x)}{\text{intercept}} & \text{if }
						E_\text{hull}(x) \le \text{intercept} \\
						0 & \text{otherwise}
					\end{cases}

			You can set the intercept in kwargs. Default intercept is 1.215
			[eV/atom]. Then, the uniqueness score with a binary distance is calculated
			as follows.

			.. math::
				U = \frac{1}{n} \sum_{i=1}^{n} V(x_i) \cdot S(x_i) \cdot I \left(
				\land_{j=1}^{i-1} \left( V(x_j) == 0 \lor S(x_j) == 0 \lor d(x_i, x_j)
				\neq 0 \right) \right),

			where :math:`\{x_1, x_2, \ldots, x_n\}` is the set of generated crystals,
			:math:`I` is the indicator function, :math:`S` is either :math:`S_b` or
			:math:`S_c`, and :math:`V` is the validity function that returns 1 for valid
			crystals and 0 for invalid crystals. For a continuous distance, the
			uniqueness score is calculated as follows.

			.. math::
				U = \frac{1}{n} \sum_{i=1}^{n} V(x_i) \cdot S(x_i) \cdot \left(
				\frac{1}{n} \sum_{j=1}^{n} V(x_j) \cdot S(x_j) \cdot d(x_i, x_j)
				\right)
				.

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		if distance not in SUPPORTED_DISTANCES:
			raise ValueError(f"Unsupported distance: {distance}.")
		if validity is not None:
			for v in validity:
				if v not in SUPPORTED_VALIDITY:
					raise ValueError(f"Unsupported validity method: {v}.")
		if stability not in [None, "binary", "continuous"]:
			raise ValueError(f"Unsupported stability criterion: {stability}.")

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
				multiprocessing,
				n_processes,
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

		# Step 2: Validity check (optional)
		valid_indices = np.ones(self.n_samples, dtype=bool)
		# Remove crystals whose embeddings could not be computed
		valid_indices &= np.array(
			[d_mtx_i0 != float("nan") for d_mtx_i0 in d_mtx[:, 0]]
		)
		if validity is not None:
			if "smact" in validity:
				valid_indices &= screen_smact(self.gen_xtals, dir_intermediate_gen)
			else:
				pass

		# Step 3: Stability evaluation (optional)
		stability_scores = np.ones(self.n_samples)  # \in [0, 1]. 1 means stable.
		if stability is not None:
			args_stability = kwargs.get("args_stability", {})
			diagram = args_stability.get("diagram", "mp_250618")
			if "args_stability" in kwargs:
				kwargs["args_stability"].pop("diagram", None)
			stability_scores = compute_stability_scores(
				self.gen_xtals,
				diagram=diagram,
				dir_intermediate=dir_intermediate_gen,
				binary=(stability == "binary"),
				**kwargs.get("args_stability", {}),
			)

		# Step 4: Compute uniqueness
		start_time_metric = time.time()
		if distance in BINARY_DISTANCES:
			is_unique = np.array(
				[
					1
					if np.all(
						np.logical_or(
							np.logical_or(
								~valid_indices[:i], stability_scores[:i] == 0
							),
							d_mtx[i, :i] != 0,
						)
					)
					else 0
					for i in range(len(d_mtx))
				]
			)
			uniqueness = (
				np.sum(valid_indices * stability_scores * is_unique) / self.n_samples
			)
		elif distance in CONTINUOUS_DISTANCES:
			vs = valid_indices * stability_scores
			uniqueness = np.sum(vs[:, np.newaxis] * vs[np.newaxis, :] * d_mtx) / (
				self.n_samples * (self.n_samples - 1)
			)
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
		validity: str | None = None,
		stability: str | None = None,
		dir_intermediate_gen: str | None = None,
		dir_intermediate_train: str | None = None,
		multiprocessing: bool = False,
		n_processes: int | None = None,
		return_time: bool = False,
		**kwargs,
	) -> float | tuple[float, dict[str, float]]:
		r"""Evaluate the novelty of a set of crystals.

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
			validity (str | None): Method to screen the crystals. Currently supported
				methods are shown in SUPPORTED_VALIDITY in constants.py.
			stability (str | None): Stability criterion for the crystals. "continuous"
				or "binary" or None.
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
			multiprocessing (bool): Whether to use multiprocessing for distance matrix
				calculation. Default is False.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing. If None, the number of logical CPU cores - 1 will be
				used. We recommend setting this argument to a smaller number than the
				number of available CPU cores to avoid memory issues. If multiprocessing
				is False, this argument is ignored. Default is None.
			return_time (bool): Whether to return the time taken for each step.
			**kwargs: Additional keyword arguments for specific distance metrics and
				thermodynamic screening. It can contain three keys: "args_emb",
				"args_dist", and "args_stability". The value of "args_emb" is a dict of
				arguments for the calculation of embeddings, the value of "args_dist" is
				a dict of arguments for the calculation of distance matrix using the
				embeddings, and the value of "args_stability" is a dict of arguments for
				the stability evaluation. For more details, please refer to the
				`tutorial notebook`_.

		Examples:
			>>> evaluator.novelty(
			...     train_xtals="mp20",
			...     distance="smat",
			...     validity=None,
			...     stability=None,
			...     dir_intermediate_gen="./intermediate",
			...     multiprocessing=True,
			...     n_processes=10,
			...     return_time=True,
			... )
			>>> (
			...     TODO,
			...     {
			...         "nov_emb_gen": TODO,
			...         "nov_emb_train": TODO,
			...         "nov_d_mtx": TODO,
			...         "nov_metric": TODO,
			...         "nov_total": TODO,
			...     },
			... )
			>>> evaluator.novelty(
			...     train_xtals=list_of_train_xtals,
			...     distance="pdd",
			...     validity=["smact"],
			...     stability="binary",
			...     dir_intermediate_gen="./intermediate",
			...     dir_intermediate_train="./intermediate_train",
			...     multiprocessing=True,
			...     n_processes=10,
			...     return_time=False,
			...     **{
			...         "args_stability": {
			...             "diagram": "mp_250618",
			...             "threshold": 0.1,
			...         },
			...     },
			... )
			>>> TODO
			>>> evaluator.novelty(
			...     train_xtals=list_of_train_xtals,
			...     distance="amd",
			...     validity=["smact"],
			...     stability="continuous",
			...     dir_intermediate_gen="./intermediate",
			...     dir_intermediate_train="./intermediate_train",
			...     multiprocessing=False,
			...     return_time=False,
			...     **{
			...         "args_emb": {"k": 200},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...         "args_stability": {
			...             "diagram": "mp_250618",
			...             "intercept": 0.2,
			...         },
			...     },
			... )
			>>> TODO

		Returns:
			float | tuple: Novelty value or a tuple containing the novelty value
			and a dictionary of time taken for each step.

		Raises:
			ValueError: If an unsupported dataset name, distance metric, or screening
				method is provided.

		Note:
			Here, I demonstrate how novelty is computed for binary/continuous distances
			with binary/continuous stability. The binary stability score, :math:`S_b(x)`
			, for each crystal :math:`x` is defined as follows.

			.. math::
				S_b(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le \text{threshold} \\
						0 & \text{otherwise}
					\end{cases}

			You can set the threshold in kwargs (see the examples). Default threshold is
			0.1 [eV/atom]. The continuous stability score, :math:`S_c(x)`, for each
			crystal :math:`x` is defined as follows.

			.. math::
				S_c(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le 0 \\
						1 - \frac{E_\text{hull}(x)}{\text{intercept}} & \text{if }
						E_\text{hull}(x) \le \text{intercept} \\
						0 & \text{otherwise}
					\end{cases}

			You can set the intercept in kwargs. Default intercept is 1.215
			[eV/atom]. Then, the novelty score with a binary distance is calculated as
			follows.

			.. math::
				N = \frac{1}{n} \sum_{i=1}^{n} V(x_i) \cdot S(x_i) \cdot I \left(
				\land_{j=1}^{m} d(x_i, y_j) \neq 0 \right),

			where :math:`\{x_1, x_2, \ldots, x_n\}` is the set of generated crystals,
			:math:`\{y_1, y_2, \ldots, y_m\}` is the set of training crystals, :math:`I`
			is the indicator function, :math:`S` is either :math:`S_b` or :math:`S_c`,
			and :math:`V` is the validity function that returns 1 for valid crystals and
			0 for invalid crystals. For a continuous distance, the novelty score is
			calculated as follows.

			.. math::
				N = \frac{1}{n} \sum_{i=1}^{n} V(x_i) \cdot S(x_i) \cdot
				\min_{j=1 \ldots m} d(x_i, y_j)

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		if isinstance(train_xtals, str) and train_xtals not in ["mp20"]:
			raise ValueError(f"Unsupported dataset name: {train_xtals}.")
		if distance not in SUPPORTED_DISTANCES:
			raise ValueError(f"Unsupported distance: {distance}.")
		if validity is not None:
			for v in validity:
				if v not in SUPPORTED_VALIDITY:
					raise ValueError(f"Unsupported validity method: {v}.")
		if stability not in [None, "binary", "continuous"]:
			raise ValueError(f"Unsupported stability criterion: {stability}.")

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
				multiprocessing,
				n_processes,
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

		# Step 2: Validity check(optional)
		valid_indices_gen = np.ones(self.n_samples, dtype=bool)
		valid_indices_train = np.ones(len(d_mtx[0]), dtype=bool)
		# Remove crystals whose embeddings could not be computed
		valid_indices_gen &= np.array(
			[d_mtx_i0 != float("nan") for d_mtx_i0 in d_mtx[:, 0]]
		)
		valid_indices_train &= np.array(
			[d_mtx_0j != float("nan") for d_mtx_0j in d_mtx[0]]
		)
		if validity is not None:
			if "smact" in validity:
				valid_indices_gen &= screen_smact(self.gen_xtals, dir_intermediate_gen)
			else:
				pass

		# Step 3: Stability evaluation (optional)
		stability_scores = np.ones(self.n_samples)  # \in [0, 1]. 1 means stable.
		if stability is not None:
			args_stability = kwargs.get("args_stability", {})
			diagram = args_stability.get("diagram", "mp_250618")
			if "args_stability" in kwargs:
				kwargs["args_stability"].pop("diagram", None)
			stability_scores = compute_stability_scores(
				self.gen_xtals,
				diagram=diagram,
				dir_intermediate=dir_intermediate_gen,
				binary=(stability == "binary"),
				**kwargs.get("args_stability", {}),
			)

		# Step 4: Compute novelty
		start_time_metric = time.time()
		if distance in BINARY_DISTANCES:
			is_novel = np.array(
				[
					1
					if np.all(np.logical_or(~valid_indices_train, d_mtx[i] != 0))
					else 0
					for i in range(len(d_mtx))
				]
			)
			novelty = (
				np.sum(valid_indices_gen * stability_scores * is_novel) / self.n_samples
			)
		elif distance in CONTINUOUS_DISTANCES:
			d_mtx[:, ~valid_indices_train] = float("inf")
			novelty = (
				np.sum(valid_indices_gen * stability_scores * np.min(d_mtx, axis=1))
				/ self.n_samples
			)
		end_time_metric = time.time()
		times["nov_metric"] = end_time_metric - start_time_metric
		times["nov_total"] = sum(times.values())

		if return_time:
			return novelty, times
		else:
			return novelty
