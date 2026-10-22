"""This module offers classes to compute the validity of crystal structures."""

import time
from abc import ABC, abstractmethod

import numpy as np
from smact.screening import smact_validity

from .crystal import Crystal


class SingleValidator(ABC):
	"""Abstract base class for single-method validity validators."""

	@abstractmethod
	def validate(self, xtals: list[Crystal]) -> np.ndarray:
		"""Validate a list of crystals.

		Args:
			xtals (list[Crystal]): List of crystals to validate.

		Returns:
			np.ndarray: Array of validity scores for each crystal.
		"""
		...


class SMACTValidator(SingleValidator):
	"""Class to calculate validity of crystal structures using SMACT."""

	def __init__(self) -> None:
		"""Initialize SMACTValidator."""
		pass

	def validate(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Validate a list of crystals using SMACT.

		Uses SMACT 4 defaults: ICSD24 oxidation states with consensus 3,
		commonality "low", and no mixed valence. Scores can differ from SMACT 3,
		which used commonality "medium".

		Args:
			xtals (list[Crystal]): List of crystals to validate.

		Returns:
			np.ndarray[float]: Array of validity scores for each crystal. A value of 1.0
			indicates that the crystal passed the SMACT screening, while 0.0 indicates
			that it failed.

		References:
			- Davies et al., (2019). SMACT: Semiconducting Materials by Analogy and
			  Chemical Theory. Journal of Open Source Software, 4(38), 1361, https://doi.org/10.21105/joss.01361
		"""
		return np.array(
			[smact_validity(xtal.get_composition_pymatgen()) for xtal in xtals],
			dtype=float,
		)


class StructureValidator(SingleValidator):
	"""Class to calculate structure-based validity of crystal structures."""

	def __init__(
		self, threshold_distance: float = 0.5, threshold_volume: float = 0.1
	) -> None:
		"""Initialize StructureValidator.

		Args:
			threshold_distance (float): Minimum allowed distance in angstroms between
				atoms, including periodic images of the same atom. Finite nonpositive
				values disable the distance check.
			threshold_volume (float): Minimum allowed unit-cell volume in cubic
				angstroms.

		References:
			- Xie et al., (2022). Crystal Diffusion Variational Autoencoder for Periodic
			  Material Generation. In International Conference on Learning
			  Representations.
		"""
		self.threshold_distance = threshold_distance
		self.threshold_volume = threshold_volume

	def validate(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Validate a list of crystals using structure-based method.

		Distances include periodic images; only a site's pair with itself in the
		same image is excluded. Distances and volumes equal to their thresholds pass.
		Nonfinite Cartesian coordinates or distance thresholds yield a score of 0.0.

		Args:
			xtals (list[Crystal]): List of crystals to validate.

		Returns:
			np.ndarray[float]: Array of validity scores for each crystal. A value of 1.0
			indicates that the crystal passed the structure-based screening, while 0.0
			indicates that it failed.
		"""
		scores = []
		for xtal in xtals:
			valid = (
				xtal.volume >= self.threshold_volume
				and np.isfinite(self.threshold_distance)
				and np.all(np.isfinite(xtal.cart_coords))
			)
			if valid and self.threshold_distance > 0:
				centers, neighbors, images, distances = xtal.get_neighbor_list(
					self.threshold_distance, exclude_self=False
				)
				distinct = (centers != neighbors) | np.any(images != 0, axis=1)
				valid = np.all(distances[distinct] >= self.threshold_distance)
			scores.append(float(valid))
		return np.array(scores, dtype=float)


class Validator:
	"""Class to calculate validity of crystal structures."""

	def __init__(self, methods: list[str], **kwargs) -> None:
		"""Initialize Validator.

		Args:
			methods (list[str]): List of validity evaluation methods to use. The
				currently supported methods are shown in SUPPORTED_VALIDITY in
				constants.py.
			**kwargs: Additional keyword arguments for each validity evaluation method.
		"""
		self.validators: dict[str, SingleValidator] = {}
		for method in methods:
			if method == "smact":
				self.validators["smact"] = SMACTValidator()
			elif method == "structure":
				self.validators["structure"] = StructureValidator(
					**kwargs.get("structure", {})
				)
			else:
				raise ValueError(f"Unsupported validity method: {method}")

	def validate(
		self,
		xtals: list[Crystal],
		skip: list[str],
	) -> tuple[dict[str, np.ndarray[float]], dict[str, float]]:
		"""Validate a list of crystals using the specified methods.

		Args:
			xtals (list[Crystal]): List of crystals to validate.
			skip (list[str]): List of validity methods to skip.

		Returns:
			tuple[dict[str, np.ndarray[float]], dict[str, float]]: A dictionary of
			individual scores from each validator, and a dictionary of time taken for
			each validity method.
		"""
		dict_individual_scores = {}
		times = {}
		for name, validator in self.validators.items():
			if name in skip:
				continue
			start_time = time.time()
			s = validator.validate(xtals)
			end_time = time.time()
			times[f"val_{name}"] = end_time - start_time
			dict_individual_scores[name] = s
		return dict_individual_scores, times
