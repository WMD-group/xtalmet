"""This module offers classes to compute the validity of crystal structures."""

import time

import numpy as np
from smact.screening import smact_validity

from .crystal import Crystal


class SingleValidator:
	"""Class to calculate validity of crystal structures using a single method."""

	def __init__(self, **kwargs) -> None:
		"""Initialize SingleValidator.

		Args:
			**kwargs: Additional keyword arguments for the validity evaluation method.
		"""
		raise NotImplementedError

	def validate(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Validate a list of crystals.

		Args:
			xtals (list[Crystal]): List of crystals to validate.

		Returns:
			np.ndarray[float]: Array of validity scores for each crystal.
		"""
		raise NotImplementedError


class SMACTValidator(SingleValidator):
	"""Class to calculate validity of crystal structures using SMACT."""

	def __init__(self) -> None:
		"""Initialize SMACTValidator."""
		pass

	def validate(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Validate a list of crystals using SMACT.

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
			threshold_distance (float): Minimum allowed distance between atoms.
			threshold_volume (float): Minimum allowed volume of the unit cell.

		References:
			- Xie et al., (2022). Crystal Diffusion Variational Autoencoder for Periodic
			  Material Generation. In International Conference on Learning
			  Representations.
		"""
		self.threshold_distance = threshold_distance
		self.threshold_volume = threshold_volume

	def validate(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Validate a list of crystals using structure-based method.

		Args:
			xtals (list[Crystal]): List of crystals to validate.

		Returns:
			np.ndarray[float]: Array of validity scores for each crystal. A value of 1.0
			indicates that the crystal passed the structure-based screening, while 0.0
			indicates that it failed.
		"""
		scores = []
		for xtal in xtals:
			dist_mat = xtal.distance_matrix
			dist_mat = dist_mat + np.diag(
				np.ones(dist_mat.shape[0]) * (self.threshold_distance + 10.0)
			)
			if (
				dist_mat.min() >= self.threshold_distance
				and xtal.volume >= self.threshold_volume
			):
				scores.append(1.0)
			else:
				scores.append(0.0)
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
