"""This module contains the Evaluator class for VSUN calculation."""

import gzip
import os
import pickle
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
from huggingface_hub import hf_hub_download
from matminer.featurizers.composition.composite import ElementProperty
from pymatgen.core import Structure

from ._cache import (
	CACHE_FORMAT_VERSION,
	MP20_REFERENCE_REVISION,
	_cache_path,
	_canonical,
	_embedding_settings,
	_fingerprint,
	_metadata,
	_read_record,
	_software_identity,
	_write_record,
)
from .constants import (
	BINARY_DISTANCES,
	DIST_WO_EMB,
	SUPPORTED_DISTANCES,
	SUPPORTED_VALIDITY,
)
from .crystal import Crystal, _to_crystal_list
from .distance import (
	_compute_embeddings,
	_embedding_mask,
	_resolve_elmd_amd_coefs,
	distance_matrix,
)
from .stability import StabilityCalculator, _validate_calibration
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
		ref_xtals: list[Crystal | Structure] | Literal["mp20"] | str | None = None,
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
			ref_xtals (list[Crystal | Structure] | Literal["mp20"] | str | None):
				Reference crystal structures (typically a training set) for novelty
				evaluation. This can be a list of crystal structures, dataset name, or a
				path to the file containing the pre-computed embeddings of the reference
				structures. If a dataset name is given, its training data will be
				downloaded from a fixed Hugging Face revision. MP20 precomputed
				embeddings are used only for default embedding settings; other settings
				emit UserWarning and compute references from MP20 training structures
				during initialization. File inputs require metadata written by
				save_reference_embeddings() with matching distance and args_emb.
				If novelty is False, this argument is
				ignored. Default is None.
			agg_func (Literal["prod", "ave"]): Aggregation function for combining V, S,
				U, and N. "prod" means multiplication, and "ave" means (weighted)
				average. Default is "prod".
			weights (dict[str, float] | None): Weights for V, S, U, and N. For "ave",
				weights are coefficients (normalized internally; equal weights if None).
				Enabled components must have a positive total weight; weights for
				disabled components do not contribute to this total.
				For "prod", weights are exponents (not normalized; default 1.0 per
				active metric if None). All supplied weights must be finite and
				non-negative. The enabled weight sum for "ave" must also be finite.
				Default is None.
			multiprocessing (bool): Whether to use multiprocessing for computing the
				embeddings of reference crystals. This argument is only effective when
				novelty is True and ref_xtals is list[Crystal | Structure] or MP20
				with non-default embedding settings. Default is False.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing. If None, the number of logical CPU cores - 1 will be
				used. We recommend setting this argument to a smaller number than the
				number of available CPU cores to avoid out-of-memory. If multiprocessing
				is False, this argument is ignored. Default is None.
			**kwargs: Additional keyword arguments. It can contain four keys:
				"args_validity", "args_stability", "args_emb", and "args_dist".
				"args_validity" is for the validity evaluation, while "args_stability"
				is for the stability evaluation. "args_emb" and "args_dist" are for the
				distance metric used in uniqueness and novelty evaluation: The former
				is for the embedding calculation, and the latter is for the distance
				matrix calculation between embeddings. For more details, please refer to
				the `tutorial notebook`_.

		Raises:
			ValueError: If novelty has an empty reference set or no valid reference
				embeddings, a reference file has missing or incompatible metadata,
				or average aggregation has zero total enabled weight.

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
			...             "mace_model": "medium-mpa-0",
			...             "intercept": 0.4289,
			...         },
			...         "args_emb": {"k": 100},
			...         "args_dist": {"metric": "chebyshev", "low_memory": False},
			...     }
			... )
			>>> # Evaluator for the SUN metric against the MP20 dataset using the
			>>> # ElMD+AMD distance, with custom kwargs.
			>>> evaluator = Evaluator(
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="elmd+amd",
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
			...             "mace_model": "medium-mpa-0",
			...             "intercept": 0.4289,
			...         },
			...         "args_emb": {"amd": {"k": 100}},
			...         "args_dist": {
			...				"elmd": {"metric": "mod_petti"},
			...				"amd": {"metric": "chebyshev", "low_memory": False},
			...				"coefs": {
			...					"elmd": float.fromhex("0x1.8d7d565a99f87p-1"),
			...					"amd": float.fromhex("0x1.ca0aa695981e5p-3")},
			...				},
			...			}
			...     }
			... )

		.. _tutorial notebook: https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb
		"""
		# Argument checks
		if not (validity is None or isinstance(validity, list)):
			raise TypeError("validity must be a list or None.")
		if validity is not None:
			for v in validity:
				if v not in SUPPORTED_VALIDITY:
					raise ValueError(f"Unsupported validity method: {v}.")
		if stability not in [None, "binary", "continuous"]:
			raise ValueError(f"Unsupported stability evaluation method: {stability}.")
		if not isinstance(uniqueness, bool):
			raise TypeError("uniqueness must be a boolean.")
		if not isinstance(novelty, bool):
			raise TypeError("novelty must be a boolean.")
		if (uniqueness or novelty) and distance not in SUPPORTED_DISTANCES:
			raise ValueError(f"Unsupported distance: {distance}.")
		if novelty and ref_xtals is None:
			raise ValueError(
				"Reference crystal structures must be provided for novelty evaluation."
			)
		if novelty and not isinstance(ref_xtals, str) and len(ref_xtals) == 0:
			raise ValueError(
				"Reference set must contain at least one reference crystal."
			)
		if novelty and not (
			ref_xtals == "mp20"
			or (type(ref_xtals) is str and Path(ref_xtals).is_file())
			or all(isinstance(xtal, (Crystal, Structure)) for xtal in ref_xtals)
		):
			raise ValueError(f"Unsupported ref_xtals: {ref_xtals}")
		if agg_func not in ["prod", "ave"]:
			raise ValueError(f"Unsupported agg_func: {agg_func}.")
		if not (weights is None or isinstance(weights, dict)):
			raise TypeError("weights must be a dictionary or None.")
		if weights is not None:
			for key in weights:
				if key not in ["validity", "stability", "uniqueness", "novelty"]:
					raise ValueError(f"Unsupported weight key: {key}.")
				if not np.isfinite(weights[key]) or weights[key] < 0.0:
					raise ValueError("Weights must be finite and non-negative.")
		if not (validity is not None or stability is not None or uniqueness or novelty):
			raise ValueError(
				"At least one of validity, stability, uniqueness, or novelty must be specified."
			)

		# Initial setup
		self.evaluate_validity = validity is not None
		self.evaluate_stability = stability is not None
		self.evaluate_uniqueness = uniqueness
		self.evaluate_novelty = novelty
		self._setup_aggregation(agg_func, weights)
		self.validator = (
			Validator(methods=validity, **kwargs.get("args_validity", {}))
			if validity is not None
			else None
		)
		self.distance = distance if uniqueness or novelty else None
		self.args_emb = kwargs.get("args_emb", {})
		self.args_dist = kwargs.get("args_dist", {})
		if self.distance == "elmd+amd":
			_resolve_elmd_amd_coefs(self.args_dist)
		if stability is not None:
			_validate_calibration(
				stability == "binary",
				**{
					key: value
					for key, value in kwargs.get("args_stability", {}).items()
					if key in ("threshold", "intercept")
				},
			)

		self._prepare_ref_xtals(
			ref_xtals, distance, multiprocessing, n_processes, **kwargs
		)
		self.stability_calculator = (
			StabilityCalculator(
				binary=(stability == "binary"), **kwargs.get("args_stability", {})
			)
			if stability is not None
			else None
		)

	def _setup_aggregation(
		self, agg_func: str, weights: dict[str, float] | None
	) -> None:
		"""Set up the aggregation function from agg_func and weights.

		Args:
			agg_func (str): Aggregation function; "prod" or "ave".
			weights (dict[str, float] | None): Per-metric weights for "ave" mode.
		"""
		self.weights: dict[str, float] = {
			"validity": 0.0,
			"stability": 0.0,
			"uniqueness": 0.0,
			"novelty": 0.0,
		}
		if self.evaluate_validity:
			self.weights["validity"] = (
				1.0 if weights is None else weights.get("validity", 1.0)
			)
		if self.evaluate_stability:
			self.weights["stability"] = (
				1.0 if weights is None else weights.get("stability", 1.0)
			)
		if self.evaluate_uniqueness:
			self.weights["uniqueness"] = (
				1.0 if weights is None else weights.get("uniqueness", 1.0)
			)
		if self.evaluate_novelty:
			self.weights["novelty"] = (
				1.0 if weights is None else weights.get("novelty", 1.0)
			)
		if agg_func == "prod":
			w = self.weights
			self.agg_func = lambda v, s, u, n: (
				v ** w["validity"]
				* s ** w["stability"]
				* u ** w["uniqueness"]
				* n ** w["novelty"]
			)
		else:  # "ave"
			total_weight = sum(self.weights.values())
			if not np.isfinite(total_weight) or total_weight <= 0:
				raise ValueError(
					"Average aggregation requires a finite positive enabled weight sum."
				)
			self.weights = {k: v / total_weight for k, v in self.weights.items()}
			w = self.weights
			self.agg_func = lambda v, s, u, n: (
				w["validity"] * v
				+ w["stability"] * s
				+ w["uniqueness"] * u
				+ w["novelty"] * n
			)

	def _prepare_ref_xtals(
		self,
		ref_xtals: list[Crystal | Structure] | Literal["mp20"] | str | None,
		distance: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		**kwargs,
	) -> None:
		"""Load or compute reference crystal embeddings for novelty evaluation.

		Args:
			ref_xtals: Reference crystals, dataset name, file path, or None.
			distance (str | None): Distance metric for embedding.
			multiprocessing (bool): Whether to use multiprocessing.
			n_processes (int | None): Maximum number of processes.
			**kwargs: Additional keyword arguments including "args_emb".
		"""
		if not self.evaluate_novelty:
			self.ref_xtals = None
			return
		if ref_xtals == "mp20":
			print("Downloading MP-20 training data from Hugging Face...")
			custom = _canonical(
				_embedding_settings(distance, self.args_emb)
			) != _canonical(_embedding_settings(distance, {}))
			if custom:
				warnings.warn(
					f"Non-default args_emb for {distance}; computing MP20 reference "
					"embeddings from training structures with the requested settings.",
					UserWarning,
					stacklevel=3,
				)
			path_embs_train = hf_hub_download(
				repo_id="masahiro-negishi/xtalmet",
				filename=f"mp20/train/train_{'smat' if custom else distance}.pkl.gz",
				repo_type="dataset",
				revision=MP20_REFERENCE_REVISION,
			)
			self.ref_xtals = self._read_pickle_gz(path_embs_train)
			if custom:
				self.ref_xtals = _compute_embeddings(
					distance,
					_to_crystal_list(self.ref_xtals),
					multiprocessing,
					n_processes,
					**self.args_emb,
				)
		elif type(ref_xtals) is str:
			try:
				record = _read_record(ref_xtals)
			except (
				OSError,
				EOFError,
				pickle.UnpicklingError,
				ValueError,
				TypeError,
			) as exc:
				raise ValueError(
					"Reference files require versioned embedding metadata. Recompute "
					"from reference structures and use save_reference_embeddings()."
				) from exc
			self.ref_xtals = record["data"]
			if not isinstance(self.ref_xtals, list):
				raise ValueError("Reference embeddings must be a list.")
			expected = {
				"kind": "reference_embeddings",
				"distance": distance,
				"args_emb": _canonical(_embedding_settings(distance, self.args_emb)),
				"count": len(self.ref_xtals),
			}
			if any(record["metadata"].get(k) != v for k, v in expected.items()):
				raise ValueError(
					"Reference embedding metadata does not match distance/args_emb."
				)
			self._check_reference_shapes()
		else:
			print("Preparing reference crystals for novelty evaluation...")
			ref_xtals_raw = _to_crystal_list(ref_xtals)
			self.ref_xtals = _compute_embeddings(
				distance=distance,
				xtals=ref_xtals_raw,
				multiprocessing=multiprocessing,
				n_processes=n_processes,
				**kwargs.get("args_emb", {}),
			)
		if len(self.ref_xtals) == 0:
			raise ValueError(
				"Reference set must contain at least one reference crystal."
			)
		if not _embedding_mask(distance, self.ref_xtals).any():
			raise ValueError("Novelty requires at least one valid reference embedding.")
		self._reference_distance = distance
		self._reference_settings = deepcopy(
			_embedding_settings(distance, self.args_emb)
		)

	def _check_reference_settings(self) -> None:
		"""Do not relabel already-prepared references after configuration mutation."""
		if self.distance != self._reference_distance or _canonical(
			_embedding_settings(self.distance, self.args_emb)
		) != _canonical(self._reference_settings):
			raise ValueError(
				"Reference embeddings were prepared with different distance/args_emb; create a new Evaluator."
			)

	def _check_reference_shapes(self) -> None:
		"""Reject feature dimensions incompatible with the declared neighbor count."""
		settings = _embedding_settings(self.distance, self.args_emb)
		magpie_size = (
			len(ElementProperty.from_preset("magpie", impute_nan=True).feature_labels())
			if self.distance == "magpie"
			else None
		)
		for emb, valid in zip(
			self.ref_xtals, _embedding_mask(self.distance, self.ref_xtals), strict=True
		):
			if not valid:
				continue
			if self.distance == "elmd+amd":
				shape, expected = np.shape(emb[1]), (settings["amd"]["k"],)
			elif self.distance == "amd":
				shape, expected = np.shape(emb), (settings["k"],)
			elif self.distance == "pdd":
				shape, expected = np.shape(emb), (len(emb), settings["k"] + 1)
			elif self.distance == "magpie":
				shape, expected = np.shape(emb), (magpie_size,)
			else:
				continue
			if shape != expected:
				raise ValueError("Reference embedding shape does not match args_emb.")

	def save_reference_embeddings(self, path: str | Path) -> None:
		"""Save prepared references with the metadata required for later reuse.

		Args:
			path (str | Path): Destination gzip-pickle file. Existing files are
				replaced.

		Raises:
			ValueError: If novelty is disabled or feature shapes contradict args_emb.
		"""
		if not self.evaluate_novelty:
			raise ValueError("No reference embeddings: novelty evaluation is disabled.")
		self._check_reference_settings()
		self._check_reference_shapes()
		_write_record(
			path,
			self.ref_xtals,
			{
				"kind": "reference_embeddings",
				"distance": self.distance,
				"args_emb": _canonical(
					_embedding_settings(self.distance, self.args_emb)
				),
				"count": len(self.ref_xtals),
				"software": _software_identity(),
			},
		)

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

	def _load_cached(
		self,
		dir_intermediate: str | None,
		filename: str,
		metadata: dict | None,
		shape: tuple[int, ...] | None = None,
		count: int | None = None,
	) -> Any:
		"""Load cached data from dir_intermediate/filename if it exists.

		Args:
			dir_intermediate (str | None): Directory to look in. If None, returns None.
			filename (str): Filename within the directory.
			metadata (dict | None): Required calculation identity, or None to skip IO.
			shape (tuple[int, ...] | None): Expected array shape.
			count (int | None): Expected embedding count.

		Returns:
			Any: Loaded data, or None if not available.
		"""
		if dir_intermediate is None or metadata is None:
			return None
		path = _cache_path(dir_intermediate, filename, metadata)
		if not path.exists():
			return None
		try:
			record = _read_record(path)
			data = record["data"]
			if record["metadata"] != metadata:
				raise ValueError("metadata mismatch")
			if shape is not None and (
				not isinstance(data, np.ndarray)
				or data.shape != shape
				or data.dtype.kind not in "fbiu"
			):
				raise ValueError("array shape/dtype mismatch")
			if count is not None and (not isinstance(data, list) or len(data) != count):
				raise ValueError("embedding count mismatch")
		except (
			OSError,
			EOFError,
			pickle.UnpicklingError,
			ValueError,
			TypeError,
			ImportError,
			AttributeError,
		) as exc:
			warnings.warn(
				f"Ignoring invalid cache {path}: {exc}", UserWarning, stacklevel=2
			)
			return None
		return data

	def _save_cached(
		self,
		dir_intermediate: str | None,
		filename: str,
		data: Any,
		metadata: dict | None,
	) -> None:
		"""Save data to dir_intermediate/filename.

		Args:
			dir_intermediate (str | None): Directory to save in. If None, no-op.
			filename (str): Filename within the directory.
			data (Any): Data to save.
			metadata (dict | None): Calculation identity, or None to skip IO.
		"""
		if dir_intermediate is None or metadata is None:
			return
		_write_record(_cache_path(dir_intermediate, filename, metadata), data, metadata)

	def _prepare_generated_embeddings(
		self,
		xtals: list[Crystal],
		dir_intermediate: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		context: dict | None,
	) -> tuple[list, np.ndarray, float]:
		"""Prepare generated embeddings once per evaluation, with or without disk IO."""
		metadata = _metadata(
			context,
			args_emb=_embedding_settings(self.distance, self.args_emb),
			distance=self.distance,
		)
		embs = self._load_cached(
			dir_intermediate, f"emb_{self.distance}.pkl.gz", metadata, count=len(xtals)
		)
		t_emb = 0.0
		if embs is None:
			if self.distance in DIST_WO_EMB:
				embs = xtals
			else:
				start = time.time()
				embs = _compute_embeddings(
					self.distance, xtals, multiprocessing, n_processes, **self.args_emb
				)
				t_emb = time.time() - start
			self._save_cached(
				dir_intermediate, f"emb_{self.distance}.pkl.gz", embs, metadata
			)
		return embs, _embedding_mask(self.distance, embs), t_emb

	def _d_mtx(
		self,
		prepared: tuple[list, np.ndarray, float],
		dir_intermediate: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		metric: Literal["uni", "nov"],
		context: dict | None,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
		"""Load or compute distances and recover masks from the source embeddings.

		Args:
			prepared (tuple): Generated embeddings, validity mask, and embedding time.
			dir_intermediate (str | None): Directory to search for pre-computed
				embeddings and distance matrix for faster computation.
			multiprocessing (bool): Whether to use multiprocessing for distance matrix
				computation.
			n_processes (int | None): Maximum number of processes to use for
				multiprocessing.
			metric (Literal["uni", "nov"]): Metric type for distance matrix computation.
			context (dict | None): Input and software identity for disk caching.

		Returns:
			tuple[np.ndarray, np.ndarray, np.ndarray, float, float]: Distance matrix,
				generated and target validity masks, embedding time, and matrix time.
		"""
		embs, valid_gen, t_emb = prepared
		target = self.ref_xtals if metric == "nov" else None
		valid_target = (
			_embedding_mask(self.distance, target) if target is not None else valid_gen
		)
		if metric == "nov" and not valid_target.any():
			raise ValueError("Novelty requires at least one valid reference embedding.")
		args = (
			self.args_dist.get("amd", {})
			if self.distance == "elmd+amd"
			else self.args_dist
		)
		metadata = (
			_metadata(
				None if "out" in args else context,
				distance=self.distance,
				args_emb=_embedding_settings(self.distance, self.args_emb),
				args_dist=self.args_dist,
				normalize=True,
				metric=metric,
				generated=_fingerprint(embs),
				reference=_fingerprint(target) if target is not None else None,
			)
			if context is not None
			else None
		)
		d_mtx = self._load_cached(
			dir_intermediate,
			f"mtx_{metric}_{self.distance}.pkl.gz",
			metadata,
			shape=(len(embs), len(target) if target is not None else len(embs)),
		)
		t_d_mtx = 0.0
		if d_mtx is None:
			result = distance_matrix(
				distance=self.distance,
				xtals_1=embs,
				xtals_2=target,
				normalize=True,
				multiprocessing=multiprocessing,
				n_processes=n_processes,
				verbose=True,
				args_emb=self.args_emb,
				args_dist=self.args_dist,
			)
			d_mtx = result[0]
			t_d_mtx = result[-1]["d_mtx"]
			self._save_cached(
				dir_intermediate,
				f"mtx_{metric}_{self.distance}.pkl.gz",
				d_mtx,
				metadata,
			)
		return d_mtx, valid_gen, valid_target, t_emb, t_d_mtx

	def _evaluate_validity(
		self,
		xtals: list[Crystal],
		n_samples: int,
		dir_intermediate: str | None,
		context: dict | None,
	) -> tuple[np.ndarray, dict[str, float]]:
		"""Compute validity scores.

		Args:
			xtals (list[Crystal]): Crystal structures to evaluate.
			n_samples (int): Number of samples.
			dir_intermediate (str | None): Directory for cached intermediate results.
			context (dict | None): Input and software identity for disk caching.

		Returns:
			tuple[np.ndarray, dict[str, float]]: Validity scores and timing dict.
		"""
		dict_individual_scores_precomputed = {}
		metadata = {
			method: _metadata(context, method=method, settings=vars(validator))
			for method, validator in self.validator.validators.items()
		}
		for method in self.validator.validators:
			data = self._load_cached(
				dir_intermediate,
				f"val_{method}.pkl.gz",
				metadata[method],
				shape=(n_samples,),
			)
			if data is not None:
				dict_individual_scores_precomputed[method] = data

		dict_individual_scores, times = self.validator.validate(
			xtals=xtals, skip=list(dict_individual_scores_precomputed.keys())
		)
		all_arrays = list(dict_individual_scores_precomputed.values()) + list(
			dict_individual_scores.values()
		)
		validity_scores = (
			np.prod(np.stack(all_arrays), axis=0)
			if all_arrays
			else np.ones(n_samples, dtype=float)
		)
		for method in dict_individual_scores_precomputed:
			times[f"val_{method}"] = 0.0
		for method, scores in dict_individual_scores.items():
			self._save_cached(
				dir_intermediate, f"val_{method}.pkl.gz", scores, metadata[method]
			)
		return validity_scores, times

	def _evaluate_stability(
		self,
		xtals: list[Crystal],
		dir_intermediate: str | None,
		context: dict | None,
	) -> tuple[np.ndarray, dict[str, float]]:
		"""Compute stability scores.

		Args:
			xtals (list[Crystal]): Crystal structures to evaluate.
			dir_intermediate (str | None): Directory for cached intermediate results.
			context (dict | None): Input and software identity for disk caching.

		Returns:
			tuple[np.ndarray, dict[str, float]]: Stability scores and timing dict.
		"""
		metadata = None
		if context is not None:
			try:
				metadata = _metadata(
					context, stability=self.stability_calculator._cache_identity()
				)
			except (TypeError, AttributeError) as exc:
				warnings.warn(
					f"Stability disk cache disabled: {exc}", UserWarning, stacklevel=2
				)
		e_above_hulls_precomputed = self._load_cached(
			dir_intermediate, "ehull.pkl.gz", metadata, shape=(len(xtals),)
		)
		stability_scores, e_above_hulls, t_stab = (
			self.stability_calculator.compute_stability_scores(
				xtals=xtals,
				e_above_hulls_precomputed=e_above_hulls_precomputed,
			)
		)
		if e_above_hulls_precomputed is None:
			self._save_cached(dir_intermediate, "ehull.pkl.gz", e_above_hulls, metadata)
		return stability_scores, {"stab": t_stab}

	def _evaluate_uniqueness(
		self,
		prepared: tuple[list, np.ndarray, float],
		n_samples: int,
		dir_intermediate: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		context: dict | None,
	) -> tuple[np.ndarray, dict[str, float]]:
		"""Compute uniqueness scores.

		Args:
			prepared (tuple): Generated embeddings, validity mask, and embedding time.
			n_samples (int): Number of samples.
			dir_intermediate (str | None): Directory for cached intermediate results.
			multiprocessing (bool): Whether to use multiprocessing.
			n_processes (int | None): Maximum number of processes.
			context (dict | None): Input and software identity for disk caching.

		Returns:
			tuple[np.ndarray, dict[str, float]]: Uniqueness scores and timing dict.
		"""
		d_mtx_uni, valid_gen, _, t_emb, t_d_mtx = self._d_mtx(
			prepared=prepared,
			dir_intermediate=dir_intermediate,
			multiprocessing=multiprocessing,
			n_processes=n_processes,
			metric="uni",
			context=context,
		)
		uniqueness_scores = np.zeros(n_samples, dtype=float)
		if self.distance in BINARY_DISTANCES:
			for i in np.flatnonzero(valid_gen):
				uniqueness_scores[i] = np.all(d_mtx_uni[i, :i][valid_gen[:i]] != 0)
		else:
			for i in np.flatnonzero(valid_gen):
				row = d_mtx_uni[i, valid_gen]
				uniqueness_scores[i] = np.sum(row[np.isfinite(row)]) / (n_samples - 1)
		return uniqueness_scores, {"uni_emb": t_emb, "uni_d_mtx": t_d_mtx}

	def _evaluate_novelty(
		self,
		prepared: tuple[list, np.ndarray, float],
		n_samples: int,
		dir_intermediate: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		context: dict | None,
	) -> tuple[np.ndarray, dict[str, float]]:
		"""Compute novelty scores.

		Args:
			prepared (tuple): Generated embeddings, validity mask, and embedding time.
			n_samples (int): Number of samples.
			dir_intermediate (str | None): Directory for cached intermediate results.
			multiprocessing (bool): Whether to use multiprocessing.
			n_processes (int | None): Maximum number of processes.
			context (dict | None): Input and software identity for disk caching.

		Returns:
			tuple[np.ndarray, dict[str, float]]: Novelty scores and timing dict.
		"""
		d_mtx_nov, valid_gen, valid_ref, t_emb, t_d_mtx = self._d_mtx(
			prepared=prepared,
			dir_intermediate=dir_intermediate,
			multiprocessing=multiprocessing,
			n_processes=n_processes,
			metric="nov",
			context=context,
		)
		novelty_scores = np.zeros(n_samples, dtype=float)
		for i in np.flatnonzero(valid_gen):
			row = d_mtx_nov[i, valid_ref]
			if self.distance not in BINARY_DISTANCES:
				row = row[np.isfinite(row)]
				if not row.size:
					continue
			novelty_scores[i] = (
				np.all(row != 0) if self.distance in BINARY_DISTANCES else np.min(row)
			)
		return novelty_scores, {"nov_emb": t_emb, "nov_d_mtx": t_d_mtx}

	def evaluate(
		self,
		xtals: list[Crystal | Structure],
		dir_intermediate: str | None = None,
		multiprocessing: bool = False,
		n_processes: int | None = None,
	) -> tuple[float, np.ndarray, dict[str, float]]:
		r"""Evaluate the given crystal structures.

		Failed embeddings (including nonfinite numerical features) have U/N scores
		of zero, without changing V/S. Binary U compares only earlier valid samples;
		N compares only valid references. Continuous U counts failed pairs as zero
		while retaining the total-sample denominator n - 1. Continuous N takes the
		minimum over valid references. Generated embeddings and their masks are
		prepared once per call and shared by U/N, even without a disk cache.
		Cached matrices still require embeddings; missing embeddings are recomputed.
		Nonfinite smat geometry is a failed embedding. Nonfinite continuous pair
		distances contribute zero to U, retaining n - 1; N uses only finite distances
		and is zero if no finite comparison remains for a generated sample.

		Args:
			xtals (list[Crystal  |  Structure]): List of crystal structures to be
				evaluated.
			dir_intermediate (str | None): Directory to search for pre-computed
				intermediate results, such as validity scores, energy above hulls,
				embeddings, and distance matrices. Versioned records are stored under
				cache/format-1/<identity>/ within this directory. Reuse requires
				matching
				ordered inputs, relevant settings, references, model/phase data, and
				software identity. Old unversioned files are ignored and preserved.
				Invalid records are recomputed. If None, no files are loaded or saved.
				It is
				recommended to set this argument when evaluating the same large set of
				crystal structures multiple times, for example trying different
				aggregation functions. Different conditions retain separate artifacts.
				Default is None.
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

		Raises:
			ValueError: If xtals is empty, continuous uniqueness has fewer than two
				samples, novelty has no valid reference embeddings, or distance/args_emb
				were changed after preparing references.

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
			>>> 0.28, np.array([...]), {"aggregation": ..., ...}
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
			>>> 0.6119424269941065, np.array([...]), {"aggregation": ..., ...}
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
			...             "mace_model": "medium-mpa-0",
			...             "intercept": 0.4289,
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
			>>> 0.019558713928249892, np.array([...]), {"aggregation": ..., ...}
			>>> # Evaluator for the SUN metric against the MP20 dataset using the
			>>> # ElMD+AMD distance, with custom kwargs.
			>>> evaluator = Evaluator(
			...     stability="continuous",
			...     uniqueness=True,
			...     novelty=True,
			...     distance="elmd+amd",
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
			...             "mace_model": "medium-mpa-0",
			...             "intercept": 0.4289,
			...         },
			...         "args_emb": {"amd": {"k": 100}},
			...         "args_dist": {
			...				"elmd": {"metric": "mod_petti"},
			...				"amd": {"metric": "chebyshev", "low_memory": False},
			...				"coefs": {
			...					"elmd": float.fromhex("0x1.8d7d565a99f87p-1"),
			...					"amd": float.fromhex("0x1.ca0aa695981e5p-3")},
			...				},
			...			}
			...     }
			... )
			>>> evaluator.evaluate(
			...     xtals=xtals,  # list[Crystal | Structure]
			...     dir_intermediate="intermediate_results/",
			...     multiprocessing=True,
			...     n_processes=10,
			... )
			>>> 0.16403383975840835, np.array([...]), {"aggregation": ..., ...}

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
			default value 0.4289 [eV/atom]. In both cases, a higher score closer to 1
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
						\text{if agg_func = "prod"} \\
						w_V V(x) + w_S S(x) + w_U U(x) + w_N N(x) &
						\text{if agg_func ="ave"}
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
		if len(xtals) == 0:
			raise ValueError("Evaluation requires at least one generated crystal.")
		if (
			self.evaluate_uniqueness
			and self.distance not in BINARY_DISTANCES
			and len(xtals) < 2
		):
			raise ValueError(
				"Continuous uniqueness requires at least two generated crystals."
			)
		if not all(isinstance(xtal, (Crystal, Structure)) for xtal in xtals):
			raise TypeError(
				"All elements in xtals must be of type Crystal or pymatgen Structure."
			)
		xtals = _to_crystal_list(xtals)
		n_samples = len(xtals)
		if self.evaluate_novelty:
			self._check_reference_settings()

		if dir_intermediate is not None:
			os.makedirs(dir_intermediate, exist_ok=True)
		context = None
		if dir_intermediate is not None:
			try:
				context = {
					"cache_format_version": CACHE_FORMAT_VERSION,
					"inputs": _fingerprint(xtals),
					"count": n_samples,
					"software": _software_identity(),
				}
			except TypeError as exc:
				warnings.warn(f"Disk cache disabled: {exc}", UserWarning, stacklevel=2)

		times: dict[str, float] = {}

		validity_scores = np.ones(n_samples, dtype=float)
		stability_scores = np.ones(n_samples, dtype=float)
		uniqueness_scores = np.ones(n_samples, dtype=float)
		novelty_scores = np.ones(n_samples, dtype=float)

		if self.evaluate_validity:
			validity_scores, v_t = self._evaluate_validity(
				xtals, n_samples, dir_intermediate, context
			)
			times.update(v_t)

		if self.evaluate_stability:
			stability_scores, s_t = self._evaluate_stability(
				xtals, dir_intermediate, context
			)
			times.update(s_t)

		if self.evaluate_uniqueness or self.evaluate_novelty:
			prepared = self._prepare_generated_embeddings(
				xtals, dir_intermediate, multiprocessing, n_processes, context
			)

		if self.evaluate_uniqueness:
			uniqueness_scores, u_t = self._evaluate_uniqueness(
				prepared,
				n_samples,
				dir_intermediate,
				multiprocessing,
				n_processes,
				context,
			)
			times.update(u_t)
			prepared = (prepared[0], prepared[1], 0.0)

		if self.evaluate_novelty:
			novelty_scores, n_t = self._evaluate_novelty(
				prepared,
				n_samples,
				dir_intermediate,
				multiprocessing,
				n_processes,
				context,
			)
			times.update(n_t)

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
