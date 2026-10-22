"""This module offers functions to compute the stability of crystal structures."""

import gzip
import hashlib
import io
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from mace.calculators import mace_mp
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester

from ._cache import _fingerprint
from .constants import HF_VERSION
from .crystal import Crystal

_DEFAULT_THRESHOLD = 0.1
_DEFAULT_INTERCEPT = 0.4289


def _validate_calibration(
	binary: bool,
	threshold: float | None = _DEFAULT_THRESHOLD,
	intercept: float | None = _DEFAULT_INTERCEPT,
) -> None:
	"""Reject invalid active calibration before preparing models or references."""
	if binary:
		if threshold is None or not np.isfinite(threshold):
			raise ValueError("Stability threshold must be finite.")
	elif intercept is None or not np.isfinite(intercept) or intercept <= 0:
		raise ValueError("Stability intercept must be finite and positive.")


def _decode_mp_entry(entry: dict) -> ComputedEntry:
	"""Decode MP entries using module paths supported before and after the split."""

	def legacy_mson(value: object) -> object:
		if isinstance(value, list):
			return [legacy_mson(item) for item in value]
		if not isinstance(value, dict):
			return value
		value = {key: legacy_mson(item) for key, item in value.items()}
		if value.get("@module") == "pymatgen.core.entries":
			value["@module"] = "pymatgen.entries.computed_entries"
		elif value.get("@module") == "pymatgen.analysis.compatibility":
			value["@module"] = "pymatgen.entries.compatibility"
		elif value.get("@module") == "pymatgen.analysis.compatibility.mixing_scheme":
			value["@module"] = "pymatgen.entries.mixing_scheme"
		elif value.get("@module") == "pymatgen.core.structure_matcher":
			value["@module"] = "pymatgen.analysis.structure_matcher"
		elif (
			value.get("@module") == "emmet.core.vasp.calculation"
			and value.get("@class") == "PotcarSpec"
		):
			# Compatibility checks expect POTCAR dictionaries, not emmet models.
			value = {
				key: item for key, item in value.items() if not key.startswith("@")
			}
		return value

	return MontyDecoder().process_decoded(legacy_mson(entry))


class StabilityCalculator:
	"""Class to calculate stability scores of crystal structures."""

	def __init__(
		self,
		diagram: Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str = "mp_250618",
		mace_model: str = "medium-mpa-0",
		binary=True,
		threshold: float = _DEFAULT_THRESHOLD,
		intercept: float = _DEFAULT_INTERCEPT,
	) -> None:
		"""Initialize StabilityCalculator.

		Args:
			diagram (Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str): A phased
				diagram to use. If "mp_250618" is specified, the diagram constructed
				using this class from the MP entries on June 18, 2025, will be used. If
				"mp" is specified, the diagram will be constructed on the spot. You can
				also pass your own diagram or a path to it. Live MP entries are checked
				with MaterialsProject2020Compatibility; the diagram uses uncorrected
				energies of the accepted GGA/GGA+U entries.
			mace_model (str): The MACE model to use for energy prediction. Default is
				"medium-mpa-0".
			binary (bool): If True, compute binary stability scores (1 for stable, 0 for
				unstable). If False, compute continuous stability scores between 0 and
				1. Default is True.
			threshold (float): Energy above hull threshold for stability in eV/atom.
				Must be finite when binary is True. Default is 0.1 eV/atom.
			intercept (float): Intercept for linear scaling of stability scores in
				eV/atom. Only used if binary is False. Default is 0.4289 eV/atom, which
				is the 99.9th percentile of the energy above hull values for the MP20
				test data.

		Raises:
			ValueError: If the active threshold is nonfinite, or the active intercept
				is nonfinite or nonpositive. Unused calibration arguments are ignored.
		"""
		_validate_calibration(binary, threshold, intercept)
		# load or construct a phase diagram
		if isinstance(diagram, PatchedPhaseDiagram):
			ppd_mp = diagram
		elif type(diagram) is str and Path(diagram).is_file():
			with gzip.open(diagram, "rb") as f:
				ppd_mp = pickle.load(f)
		elif diagram == "mp_250618":
			path = hf_hub_download(
				repo_id="masahiro-negishi/xtalmet",
				filename="phase-diagram/ppd-mp_all_entries_uncorrected_250618.pkl.gz",
				repo_type="dataset",
				revision=HF_VERSION,
			)
			with gzip.open(path, "rb") as f:
				ppd_mp = pickle.load(f)
		elif diagram == "mp":
			MP_API_KEY = os.getenv("MP_API_KEY")
			mpr = MPRester(MP_API_KEY)
			response = mpr.request(
				"materials/thermo/?_fields=entries&formula=", mp_decode=False
			)
			all_entries = []
			for dct in response:
				all_entries.extend(_decode_mp_entry(e) for e in dct["entries"].values())
			del response
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore", message="Failed to guess oxidation states.*"
				)
				all_entries = MaterialsProject2020Compatibility().process_entries(
					all_entries, clean=True
				)
			all_entries = list(set(all_entries))  # remove duplicates
			all_entries = [
				e for e in all_entries if e.data["run_type"] in ["GGA", "GGA_U"]
			]  # Only use entries computed with GGA or GGA+U
			all_entries_uncorrected = [
				PDEntry(composition=e.composition, energy=e.uncorrected_energy)
				for e in all_entries
			]
			ppd_mp = PatchedPhaseDiagram(all_entries_uncorrected)
		else:
			raise ValueError(f"Unsupported diagram: {diagram}")
		# prepare mace model
		if mace_model == "mh-1":
			calculator = mace_mp(
				model="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
				default_dtype="float64",
				head="omat_pbe",
			)
		else:
			calculator = mace_mp(model=mace_model, default_dtype="float64")
		self.ppd_mp = ppd_mp
		self.calculator = calculator
		self.binary = binary
		self.threshold = threshold
		self.intercept = intercept

	def _cache_identity(self) -> dict:
		"""Identify actual energy inputs independently of score calibration."""
		if type(self.ppd_mp) is not PatchedPhaseDiagram or any(
			type(pd) is not PhaseDiagram for pd in self.ppd_mp.pds.values()
		):
			raise TypeError("custom phase-diagram classes cannot be fingerprinted")
		entry_ids = {}

		def entry_id(entry):
			if id(entry) not in entry_ids:
				entry_ids[id(entry)] = _fingerprint(
					(entry.composition.as_dict(), entry.energy)
				)
			return entry_ids[id(entry)]

		calculator = self.calculator
		if any(
			not type(module).__module__.startswith(
				(
					"mace.modules.",
					"e3nn.",
					"torch.nn.modules.",
					"torch.jit.",
					"torch.fx.graph_module",
				)
			)
			for model in calculator.models
			for module in model.modules()
		):
			raise TypeError("custom model classes cannot be fingerprinted")
		# MACE also uses non-tensor settings omitted from state_dict and repr.
		with io.BytesIO() as checkpoint:
			try:
				torch.save(calculator.models, checkpoint)
			except (RuntimeError, pickle.PicklingError, TypeError) as exc:
				raise TypeError(
					"model cannot be serialized for fingerprinting"
				) from exc
			models = hashlib.sha256(checkpoint.getbuffer()).hexdigest()
		return {
			"model": models,
			"calculator": {
				name: str(getattr(calculator, name))
				for name in (
					"head",
					"model_type",
					"device",
					"energy_units_to_eV",
					"length_units_to_A",
				)
			},
			"diagram": _fingerprint(
				sorted(entry_id(entry) for entry in self.ppd_mp.all_entries)
			),
			"elements": [str(element) for element in self.ppd_mp.elements],
			"element_refs": {
				str(element): entry_id(entry)
				for element, entry in self.ppd_mp.el_refs.items()
			},
			"stable_entries": [
				entry_id(entry) for entry in self.ppd_mp._stable_entries
			],
			"stable_spaces": [
				sorted(str(element) for element in space)
				for space in self.ppd_mp._stable_spaces
			],
			"patches": [
				(
					sorted(str(element) for element in space),
					_fingerprint(
						{
							"elements": [str(element) for element in pd.elements],
							"entries": [entry_id(entry) for entry in pd.qhull_entries],
							"data": pd.qhull_data,
							"facets": pd.facets,
							"simplexes": [simplex.coords for simplex in pd.simplexes],
						}
					),
				)
				for space, pd in self.ppd_mp.pds.items()
			],
		}

	def _ehull(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Compute energy above hull for a list of crystals.

		Hull queries use elemental composition, ignoring oxidation-state labels
		without mutating the input structure. Nonfinite geometry, model energies or
		hull energies produce NaN. Finite negative hull energies remain valid.

		Args:
		    xtals (list[Crystal]): List of crystals to compute energy above hull for.

		Returns:
		   np.ndarray[float]: Array of energy above hull for each crystal.
		"""
		e_above_hulls = np.full(len(xtals), np.nan)
		for idx, xtal in enumerate(xtals):
			try:
				if not (
					np.isfinite(xtal.lattice.matrix).all()
					and np.isfinite(xtal.cart_coords).all()
				):
					continue
				mace_energy = self.calculator.get_potential_energy(xtal.get_ase_atoms())
				if not np.isfinite(mace_energy):
					continue
				gen_entry = ComputedEntry(
					xtal.get_composition_pymatgen().element_composition, mace_energy
				)
				ehull = self.ppd_mp.get_e_above_hull(gen_entry, allow_negative=True)
				if np.isfinite(ehull):
					e_above_hulls[idx] = ehull
			except ValueError:
				e_above_hulls[idx] = np.nan
		return e_above_hulls

	def compute_stability_scores(
		self,
		xtals: list[Crystal],
		e_above_hulls_precomputed: np.ndarray[float] | None = None,
	) -> tuple[np.ndarray[float], np.ndarray[float], float]:
		"""Compute stability scores for a list of crystals.

		NaN and infinite energies receive zero stability, including precomputed
		energies. Finite negative energies receive stability one. Precomputed
		arrays are returned unchanged.

		Args:
			xtals (list[Crystal]): List of crystals to compute stability scores for.
			e_above_hulls_precomputed (np.ndarray[float] | None): Precomputed energy
				above hull values. If None, they will be computed internally.

		Returns:
			tuple[np.ndarray[float], np.ndarray[float], float]: A tuple of stability
			scores, raw energy above hull values, and computation time in seconds.
		"""
		start_time = time.time()
		if e_above_hulls_precomputed is None:
			e_above_hulls = self._ehull(xtals)
		else:
			e_above_hulls = e_above_hulls_precomputed
		stability_scores = np.zeros(len(xtals), dtype=float)
		finite = np.isfinite(e_above_hulls)
		if self.binary:
			stability_scores[finite & (e_above_hulls <= self.threshold)] = 1.0
		else:
			stability_scores[finite] = np.clip(
				1 - e_above_hulls[finite] / self.intercept, 0, 1
			)
		end_time = time.time()
		return stability_scores, e_above_hulls, end_time - start_time
