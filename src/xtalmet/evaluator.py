"""This module contains the Evaluator class for VSUN calculation."""

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from huggingface_hub import hf_hub_download
from pymatgen.core import Structure

from .constants import (
	BINARY_DISTANCES,
	HF_VERSION,
	SUPPORTED_DISTANCES,
	SUPPORTED_VALIDITY,
)
from .crystal import Crystal
from .distance import _compute_embeddings, distance_matrix
from .stability import StabilityCalculator
from .validity import Validator


class Evaluator:
	"""Class for evaluating a set of crystals.

	The evaluation is based on a chosen combination of validity (V), stability (S),
	uniqueness (U), and novelty (N).
	"""

	def __init__(
		self,
		validity: list[str] | None = None,
		stability: Literal["continuous", "binary", None] = None,
		uniqueness: bool = False,
		novelty: bool = False,
		distance: str | None = None,
		ref_xtals: list[Crystal | Structure] | str | None = None,
		agg_func: Literal["prod", "ave"] = "prod",
		weights: dict[str, float] | None = None,
		multiprocessing: bool = False,
		n_processes: int | None = None,
		**kwargs,
	) -> None:
		"""Initialize the Evaluator.

		Args:
			validity (list[str] | None): Approaches to evaluating validity.
				The currently supported methods are shown in SUPPORTED_VALIDITY in
				constants.py. If set to None, validity is not evaluated. Default is
				None.
			stability (Literal["continuous", "binary", None]): Stability evaluation
				method. "continuous" or "binary" or None. If set to None, stability is
				not evaluated. Default is None.
			uniqueness (bool): Whether to evaluate uniqueness. Default is False.
			novelty (bool): Whether to evaluate novelty. Default is False.
			distance (str | None): Distance metric used for uniqueness and novelty
				evaluation. The currently supported distances are listed in
				SUPPORTED_DISTANCES in constants.py. For more detailed information about
				each distance metric, please refer to the `tutorial notebook`_. If both
				uniqueness and novelty are False, this argument is ignored. Default is
				None.
			ref_xtals (list[Crystal  |  Structure]  |  str  |  None): Reference crystal
				structures (typically a training set) for novelty evaluation. This can
				be a list of crystal structures, dataset name, or a path to the file
				containing the pre-computed embeddings of the reference structures. If a
				dataset name is given, its training data will be downloaded from Hugging
				Face. If novelty is False, this argument is ignored. Default is None.
			agg_func (Literal["prod", "ave"]): Aggregation function for combining V, S,
				U, and N. "prod" means multiplication, and "ave" means (weighted)
				average. Default is "prod".
			weights (dict[str, float] | None): Weights for V, S, U, and N when agg_func
				is "ave". For example, {"validity": 0.2, "stability": 0.3, "uniqueness":
				0.2, "novelty": 0.3}. You only need to provide weights for the metrics
				you choose to evaluate. If the weights are not normalized, they will be
				normalized internally. If None, equal weights are used. Default is None.
			multiprocessing (bool): Whether to use multiprocessing for computing the
				embeddings of reference crystals. This argument is only effective when
				novelty is True and ref_xtals is list[Crystal | Structure]. Default
				is False.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing. If None, the number of logical CPU cores - 1 will be
				used. We recommend setting this argument to a smaller number than the
				number of available CPU cores to avoid out-of-memory. If multiprocessing
				is False, this argument is ignored. Default is None.
			**kwargs: Additional keyword arguments. It can contain four keys:
				"args_validiity", "args_stability", "args_emb", and "args_dist".
				"args_validity" is for the validity evaluation, while "args_stability"
				is for the stability evaluation. "args_emb" and "args_dist" are for the
				distance metric used in uniqueness and novelty evaluation: The former
				is for the embedding calculation, and the latter is for the distance
				matrix calculation between embeddings. For more details, please refer to
				the `tutorial notebook`_.

		Examples:
			>>> # Evaluator for the conventional SUN metric against the MP20 dataset
			>>> # using the StructureMatcher distance.
			>>> evaluator = Evaluator(
			...     stability="binary",
			...		uniqueness=True,
			...		novelty=True,
			...		distance="smat",
			...		ref_xtals="mp20",
			...		agg_func="prod",
			... )
			>>> # Evaluator for the VSUN metric against a custom reference dataset using
			>>> # the ElMD distance, with average aggregation.
			>>> evaluator = Evaluator(
			...     validity=["smact", "structure"],
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="elmd",
			...     ref_xtals=ref_xtals,  # list[Crystal | Structure]
			...     agg_func="ave",
			...     weights={
			...         "validity": 0.25,
			...         "stability": 0.25,
			...         "uniqueness": 0.25,
			...         "novelty": 0.25,
			...     },
			...     multiprocessing=True,
			...     n_processes=10,
			... )
			>>> # Evaluator for the VSUN metric against the MP20 dataset using the AMD
			>>> # distance, with custom kwargs.
			>>> evaluator = Evaluator(
			...     validity=["smact", "structure"],
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="amd",
			...     ref_xtals="mp20",
			...     agg_func="prod",
			...     {
			...         "args_validity": {
			...             "structure": {
			...                 "threshold_distance": 0.5,
			...                 "threshold_volume": 0.1,
			...             }
			...         },
			...         "args_stability": {
			...             "diagram": "mp_250618",
			...             "mace_model": "mace-mh-1",
			...             "intercept": 1.215,
			...         },
			...         "args_emb": {"k": 100},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...     }
			... )

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		# Argument checks
		assert validity is None or isinstance(validity, list), (
			"validity must be a list or None."
		)
		if validity is not None:
			for v in validity:
				assert v in SUPPORTED_VALIDITY, f"Unsupported validity method: {v}."
		assert stability in [None, "binary", "continuous"], (
			f"Unsupported stability evaluation method: {stability}."
		)
		assert type(uniqueness) is bool, "uniqueness must be a boolean."
		assert type(novelty) is bool, "novelty must be a boolean."
		if uniqueness or novelty:
			assert distance in SUPPORTED_DISTANCES, f"Unsupported distance: {distance}."
		if novelty:
			assert ref_xtals is not None, (
				"Reference crystal structures must be provided for novelty evaluation."
			)
			assert (
				ref_xtals == "mp20"
				or (type(ref_xtals) is str and Path(ref_xtals).is_file())
				or all(isinstance(xtal, (Crystal, Structure)) for xtal in ref_xtals)
			), f"Unsupported ref_xtals: {ref_xtals}"
		assert agg_func in ["prod", "ave"], f"Unsupported agg_func: {agg_func}."
		if agg_func == "ave":
			assert weights is None or isinstance(weights, dict), (
				"weights must be a dictionary or None."
			)
			if weights is not None:
				for key in weights:
					assert key in ["validity", "stability", "uniqueness", "novelty"], (
						f"Unsupported weight key: {key}."
					)
					assert weights[key] >= 0.0, "Weights must be non-negative."
		assert validity is not None or stability is not None or uniqueness or novelty, (
			"At least one of validity, stability, uniqueness, or novelty must be specified."
		)

		# Initial setup
		self.evaluate_validity = validity is not None
		self.validator = (
			Validator(methods=validity, **kwargs.get("args_validity", {}))
			if validity is not None
			else None
		)
		self.evaluate_stability = stability is not None
		self.stability_calculator = (
			StabilityCalculator(
				binary=(stability == "binary"), **kwargs.get("args_stability", {})
			)
			if stability is not None
			else None
		)
		self.evaluate_uniqueness = uniqueness
		self.evaluate_novelty = novelty
		self.distance = distance if uniqueness or novelty else None
		if novelty:
			if ref_xtals == "mp20":
				print("Downloading MP-20 training data from Hugging Face...")
				path_embs_train = hf_hub_download(
					repo_id="masahiro-negishi/xtalmet",
					filename=f"mp20/train/train_{distance}.pkl.gz",
					repo_type="dataset",
					revision=HF_VERSION,
				)
				self.ref_xtals = self._read_pickle_gz(path_embs_train)
			elif type(ref_xtals) is str and Path(ref_xtals).is_file():
				self.ref_xtals = self._read_pickle_gz(ref_xtals)
			else:
				print("Preparing reference crystals for novelty evaluation...")
				ref_xtals_raw = [
					xtal if isinstance(xtal, Crystal) else Crystal.from_Structure(xtal)
					for xtal in ref_xtals
				]
				self.ref_xtals = _compute_embeddings(
					distance=distance,
					xtals=ref_xtals_raw,
					multiprocessing=multiprocessing,
					n_processes=n_processes,
					**kwargs.get("args_emb", {}),
				)
		else:
			self.ref_xtals = None
		if agg_func == "prod":
			self.agg_func = lambda v, s, u, n: v * s * u * n
		else:
			self.weights: dict[str, float] = {
				"validity": 0.0,
				"stability": 0.0,
				"uniqueness": 0.0,
				"novelty": 0.0,
			}
			if self.evaluate_validity:
				self.weights["validity"] = (
					1.0 if weights is None else weights["validity"]
				)
			if self.evaluate_stability:
				self.weights["stability"] = (
					1.0 if weights is None else weights["stability"]
				)
			if self.evaluate_uniqueness:
				self.weights["uniqueness"] = (
					1.0 if weights is None else weights["uniqueness"]
				)
			if self.evaluate_novelty:
				self.weights["novelty"] = 1.0 if weights is None else weights["novelty"]
			total_weight = sum(self.weights.values())
			self.weights = {k: v / total_weight for k, v in self.weights.items()}
			self.agg_func = (
				lambda v, s, u, n: self.weights["validity"] * v
				+ self.weights["stability"] * s
				+ self.weights["uniqueness"] * u
				+ self.weights["novelty"] * n
			)
		self.args_emb = kwargs.get("args_emb", {})
		self.args_dist = kwargs.get("args_dist", {})

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

	def _d_mtx(
		self,
		xtals: list[Crystal],
		dir_intermediate: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		metric: Literal["uni", "nov"],
	) -> tuple[np.ndarray, float, float]:
		"""Compute distance matrix for uniqueness calculation.

		Args:
			xtals (list[Crystal]): List of crystal structures to compute distance matrix
				for.
			dir_intermediate (str | None): Directory to search for pre-computed
				embeddings and distance matrix for faster computation.
			multiprocessing (bool): Whether to use multiprocessing for distance matrix
				computation.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing.
			metric (Literal["uni", "nov"]): Metric type for distance matrix computation.

		Returns:
			tuple[np.ndarray, float, float]: Distance matrix, time taken for embedding
				computation, and time taken for distance matrix computation.
		"""
		times = {}
		# Use pre-computed distance matrix if available
		if dir_intermediate is not None and os.path.exists(
			os.path.join(dir_intermediate, f"mtx_{metric}_{self.distance}.pkl.gz")
		):
			d_mtx = self._read_pickle_gz(
				os.path.join(dir_intermediate, f"mtx_{metric}_{self.distance}.pkl.gz")
			)
			times[f"{metric}_emb"] = 0.0
			times[f"{metric}_d_mtx"] = 0.0
		else:
			# Prepare generated samples
			if dir_intermediate is not None and os.path.exists(
				os.path.join(dir_intermediate, f"emb_{self.distance}.pkl.gz")
			):
				xtals = self._read_pickle_gz(
					os.path.join(dir_intermediate, f"emb_{self.distance}.pkl.gz")
				)
				embed = True
			else:
				embed = False
			# Distance matrix computation
			if metric == "uni":
				d_mtx, embs, times_matrix = distance_matrix(
					distance=self.distance,
					xtals_1=xtals,
					xtals_2=None,
					normalize=True,
					multiprocessing=multiprocessing,
					n_processes=n_processes,
					verbose=True,
					**{
						"args_emb": self.args_emb,
						"args_dist": self.args_dist,
					},
				)
			else:
				d_mtx, embs, _, times_matrix = distance_matrix(
					distance=self.distance,
					xtals_1=xtals,
					xtals_2=self.ref_xtals,
					normalize=True,
					multiprocessing=multiprocessing,
					n_processes=n_processes,
					verbose=True,
					**{
						"args_emb": self.args_emb,
						"args_dist": self.args_dist,
					},
				)
			# Record times
			times[f"{metric}_emb"] = times_matrix["emb_1"]
			times[f"{metric}_d_mtx"] = times_matrix["d_mtx"]
			# Save intermediate results
			if dir_intermediate is not None:
				self._write_pickle_gz(
					d_mtx,
					os.path.join(
						dir_intermediate, f"mtx_{metric}_{self.distance}.pkl.gz"
					),
				)
				if not embed:
					self._write_pickle_gz(
						embs,
						os.path.join(dir_intermediate, f"emb_{self.distance}.pkl.gz"),
					)
		return d_mtx, times[f"{metric}_emb"], times[f"{metric}_d_mtx"]

	def evaluate(
		self,
		xtals: list[Crystal | Structure],
		dir_intermediate: str | None = None,
		multiprocessing: bool = False,
		n_processes: int | None = None,
	) -> tuple[float, np.ndarray, dict[str, float]]:
		r"""Evaluate the given crystal structures.

		Args:
			xtals (list[Crystal  |  Structure]): List of crystal structures to be
				evaluated.
			dir_intermediate (str | None): Directory to search for pre-computed
				intermediate results, such as validity scores, energy above hulls,
				embeddings, and distance matrices. If pre-computed files do not exist in
				the directory, they will be computed and saved to the directory for
				future use. If set to None, no files will be loaded or saved. It is
				recommended to set this argument when evaluating the same large set of
				crystal structures multiple times, for example trying different
				aggregation functions. The intermediate results can be shared as long as
				the same set of crystals is evaluated. Default is None.
			multiprocessing (bool): Whether to use multiprocessing for embedding and
				distance matrix computation. This argument is only effective when
				uniqueness or novelty evaluation is enabled. Default is False.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing. If None, the number of logical CPU cores - 1 will be
				used. We recommend setting this argument to a smaller number than the
				number of available CPU cores to avoid out-of-memory. If multiprocessing
				is False, this argument is ignored. Default is None.

		Returns:
			tuple[float, np.ndarray, dict[str, float]]: A tuple containing the overall
			score (float), individual scores for each crystal structure, and a
			dictionary of computation times for each evaluation component.

		Examples:
			>>> # Evaluate the conventional SUN metric using the StructureMatcher
			>>> # distance ("smat") against the MP20 dataset.
			>>> evaluator = Evaluator(
			...     stability="binary",
			...		uniqueness=True,
			...		novelty=True,
			...		distance="smat",
			...		ref_xtals="mp20",
			...		agg_func="prod",
			... )
			>>> evaluator.evaluate(
			...     xtals=xtals,  # list[Crystal | Structure]
			...     dir_intermediate="intermediate_results/",
			...     multiprocessing=True,
			...     n_processes=10,
			... )
			>>> 0.0316, np.array([...]), {"aggregation": ..., ...}
			>>> # Evaluate the VSUN metric using the ElMD distance against a custom
			>>> # reference dataset, with average aggregation.
			>>> evaluator = Evaluator(
			...     validity=["smact", "structure"],
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="elmd",
			...     ref_xtals=ref_xtals,  # list[Crystal | Structure]
			...     agg_func="ave",
			...     weights={
			...         "validity": 0.25,
			...         "stability": 0.25,
			...         "uniqueness": 0.25,
			...         "novelty": 0.25,
			...     },
			...     multiprocessing=True,
			...     n_processes=10,
			... )
			>>> evaluator.evaluate(
			...     xtals=xtals,  # list[Crystal | Structure]
			...     dir_intermediate=None,
			...     multiprocessing=False,
			...     n_processes=None,
			... )
			>>> 0.017467859079216795, np.array([...]), {"aggregation": ..., ...}
			>>> # Evaluate the VSUN metric using the AMD distance against the MP20
			>>> # dataset, with custom kwargs.
			>>> evaluator = Evaluator(
			...     validity=["smact", "structure"],
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="amd",
			...     ref_xtals="mp20",
			...     agg_func="prod",
			...     {
			...         "args_validity": {
			...             "structure": {
			...                 "threshold_distance": 0.5,
			...                 "threshold_volume": 0.1,
			...             }
			...         },
			...         "args_stability": {
			...             "diagram": "mp_250618",
			...             "mace_model": "mace-mh-1",
			...             "intercept": 1.215,
			...         },
			...         "args_emb": {"k": 100},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...     }
			... )
			>>> evaluator.evaluate(
			...     xtals=xtals,  # list[Crystal | Structure]
			...     dir_intermediate="intermediate_results/",
			...     multiprocessing=True,
			...     n_processes=10,
			... )
			>>> 0.0268044218595296, np.array([...]), {"aggregation": ..., ...}

		Note:
			Here, I demonstrate how VSUN (or its subsets) is computed. Validity
			:math:`V(x)` of each crystal :math:`x` is 0 (invalid) or 1 (valid) depending
			on whether it passes the specified validity checks. If you specify multiple
			validity methods :math:`V_1(x)`, :math:`V_2(x)`, ..., the overall validity
			is the product of individual validity scores:

			.. math::
				V(x) = \prod_i V_i(x)

			Stability :math:`S(x)` is computed from the energy above hull, and has two
			variants: binary and continuous. The binary stability score :math:`S_b(x)`
			is defined as follows.

			.. math::
				S_b(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le \text{threshold} \\
						0 & \text{otherwise}
					\end{cases}

			"threshold" can be specified in kwargs when initializing the Evaluator. The
			default value is 0.1 [eV/atom]. The continuous stability score
			:math:`S_c(x)` is defined as follows.

			.. math::
				S_c(x) =
					\begin{cases}
						1 & \text{if } E_\text{hull}(x) \le 0 \\
						1 - \frac{E_\text{hull}(x)}{\text{intercept}} & \text{if }
						0 \le E_\text{hull}(x) \le \text{intercept} \\
						0 & \text{otherwise}
					\end{cases}

			"intercept" can be specified in kwargs when initializing the Evaluator. The
			default value is 1.215 [eV/atom]. In both cases, a higher score closer to 1
			indicates a more stable structure. The definition of uniqueness :math:`U(x)`
			depends on the chosen distance metric. For a binary distance :math:`d_b`,
			uniqueness of the i-th crystal :math:`x_i` in the set of crystals
			:math:`\{x_1, x_2, \ldots, x_n\}` is defined as follows.

			.. math::
				U_b(x_i) = I \left(\land_{j=1}^{i-1} d_b(x_i, x_j) \neq 0 \right),

			where :math:`I` is the indicator function. For a continuous distance
			:math:`d_c`, uniqueness is defined as follows.

			.. math::
				U_c(x_i) = \frac{1}{n-1} \sum_{j=1}^{n} d_c(x_i, x_j).

			In both cases, the score ranges from 0 to 1, since the binary distance takes
			values of 0 or 1, and the continuous distance is normalized to be between 0
			and 1. A higher score indicates a more unique structure. Novelty
			:math:`N(x)` is defined similarly to uniqueness, but against a reference set
			of crystals. For a binary distance :math:`d_b`, novelty of the i-th crystal
			:math:`x_i` is defined as follows.

			.. math::
				N_b(x_i) = I \left(\land_{j=1}^{m} d_b(x_i, y_j) \neq 0 \right),

			where :math:`\{y_1, y_2, \ldots, y_m\}` is the reference set of crystals.
			For a continuous distance :math:`d_c`, novelty is defined as follows.

			.. math::
				N_c(x_i) = \min_{j=1 \ldots m} d_c(x_i, y_j).

			In both cases, the socre ranges from 0 to 1, and a higher score indicates a
			more novel structure. Finally, the VSUN score of each crystal :math:`x` is
			computed by aggregating the individual scores using either multiplication or
			(weighted) average:

			.. math::
				\text{VSUN}(x) =
					\begin{cases}
						V(x) S(x) U(x) N(x) &
						\text{if agg_func = “prod"} \\
						w_V V(x) + w_S S(x) + w_U U(x) + w_N N(x) &
						\text{if agg_func =“ave"}
					\end{cases}

			where :math:`w_V`, :math:`w_S`, :math:`w_U`, and :math:`w_N` are the
			normalized weights for validity, stability, uniqueness, and novelty,
			respectively. If only a subset of VSUN is evaluated, each crystal's score is
			computed using only the specified components in the similar manner. The
			overall single score for the entire set of crystals is then obtained by
			averaging the individual scores.

			.. math::
				\text{VSUN} = \frac{1}{n} \sum_{i=1}^{n} \text{VSUN}(x_i)
		"""
		assert all(isinstance(xtal, (Crystal, Structure)) for xtal in xtals), (
			"All elements in xtals must be of type Crystal or pymatgen Structure."
		)
		xtals = [
			xtal if isinstance(xtal, Crystal) else Crystal.from_Structure(xtal)
			for xtal in xtals
		]
		n_samples = len(xtals)

		if dir_intermediate is not None:
			os.makedirs(dir_intermediate, exist_ok=True)

		times: dict[str, float] = {}

		# Validity
		if self.evaluate_validity:
			# Load pre-computed validity scores if available
			dict_individual_scores_precomputed = {}
			if dir_intermediate is not None:
				for method in self.validator.validators:
					path = os.path.join(dir_intermediate, f"val_{method}.pkl.gz")
					if os.path.exists(path):
						dict_individual_scores_precomputed[method] = (
							self._read_pickle_gz(path)
						)
						times[f"val_{method}"] = 0.0
			# Compute validity scores for the rest
			dict_individual_scores, times_validity = self.validator.validate(
				xtals=xtals, skip=list(dict_individual_scores_precomputed.keys())
			)
			validity_scores = np.ones(n_samples, dtype=float)
			for val in dict_individual_scores_precomputed.values():
				validity_scores *= val
			for val in dict_individual_scores.values():
				validity_scores *= val
			for key, val in times_validity.items():
				times[key] = val
			# Save validity scores
			if dir_intermediate is not None:
				for method, scores in dict_individual_scores.items():
					path = os.path.join(dir_intermediate, f"val_{method}.pkl.gz")
					self._write_pickle_gz(scores, path)
		else:
			validity_scores = np.ones(n_samples, dtype=float)

		# Stability
		if self.evaluate_stability:
			# Load pre-computed e_above_hulls if available
			if dir_intermediate is not None and os.path.exists(
				os.path.join(dir_intermediate, "ehull.pkl.gz")
			):
				e_above_hulls_precomputed = self._read_pickle_gz(
					os.path.join(dir_intermediate, "ehull.pkl.gz")
				)
			else:
				e_above_hulls_precomputed = None
			# Compute stability scores
			stability_scores, e_above_hulls, times["stab"] = (
				self.stability_calculator.compute_stability_scores(
					xtals=xtals,
					e_above_hulls_precomputed=e_above_hulls_precomputed,
				)
			)
			# Save e_above_hulls values
			if dir_intermediate is not None and e_above_hulls_precomputed is None:
				self._write_pickle_gz(
					e_above_hulls,
					os.path.join(dir_intermediate, "ehull.pkl.gz"),
				)
		else:
			stability_scores = np.ones(n_samples, dtype=float)

		# Uniqueness
		if self.evaluate_uniqueness:
			d_mtx_uni, times["uni_emb"], times["uni_d_mtx"] = self._d_mtx(
				xtals=xtals,
				dir_intermediate=dir_intermediate,
				multiprocessing=multiprocessing,
				n_processes=n_processes,
				metric="uni",
			)
			# Crystals are invalid if their embeddings could not be computed
			validity_scores *= np.array(
				[d_mtx_i0 != float("nan") for d_mtx_i0 in d_mtx_uni[:, 0]], dtype=float
			)
			# Compute uniqueness scores
			if self.distance in BINARY_DISTANCES:
				uniqueness_scores = np.array(
					[np.all(d_mtx_uni[i, :i] != 0) for i in range(n_samples)],
					dtype=float,
				)
			else:
				uniqueness_scores = np.sum(d_mtx_uni, axis=1) / (n_samples - 1)
		else:
			uniqueness_scores = np.ones(n_samples, dtype=float)

		# Novelty
		if self.evaluate_novelty:
			d_mtx_nov, times["nov_emb"], times["nov_d_mtx"] = self._d_mtx(
				xtals=xtals,
				dir_intermediate=dir_intermediate,
				multiprocessing=multiprocessing,
				n_processes=n_processes,
				metric="nov",
			)
			validity_ref = np.array(
				[d_mtx_0j != float("nan") for d_mtx_0j in d_mtx_nov[0]]
			)
			if self.distance in BINARY_DISTANCES:
				novelty_scores = np.array(
					[
						np.all(np.logical_or(~validity_ref, d_mtx_nov[i] != 0))
						for i in range(n_samples)
					],
					dtype=float,
				)
			else:
				d_mtx_nov[:, ~validity_ref] = float("inf")
				novelty_scores = np.min(d_mtx_nov, axis=1)
		else:
			novelty_scores = np.ones(n_samples, dtype=float)

		# Aggregation
		start_time_metric = time.time()
		individual_scores = self.agg_func(
			validity_scores, stability_scores, uniqueness_scores, novelty_scores
		)
		overall_score = np.mean(individual_scores)
		end_time_metric = time.time()
		times["aggregation"] = end_time_metric - start_time_metric
		times["total"] = sum(times.values())

		return overall_score, individual_scores, times
