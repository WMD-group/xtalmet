"""This module offers functions to compute the stability of crystal structures."""

import gzip
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import hf_hub_download
from mace.calculators import mace_mp
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester

from .constants import HF_VERSION
from .crystal import Crystal


class StabilityCalculator:
	"""Class to calculate stability scores of crystal structures."""

	def __init__(
		self,
		diagram: Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str = "mp_250618",
		mace_model: str = "mace-mh-1",
		binary=True,
		threshold: float = 0.1,
		intercept: float = 1.215,
	) -> None:
		"""Initialize StabilityCalculator.

		Args:
			diagram (Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str): A phased
				diagram to use. If "mp_250618" is specified, the diagram constructed
				using this class from the MP entries on June 18, 2025, will be used. If
				"mp" is specified, the diagram will be constructed on the spot. You can
				also pass your own diagram or a path to it.
			mace_model (str): The MACE model to use for energy prediction. Default is
				"mace-mh-1".
			binary (bool): If True, compute binary stability scores (1 for stable, 0 for
				unstable). If False, compute continuous stability scores between 0 and
				1. Default is True.
			threshold (float): Energy above hull threshold for stability in eV/atom.
				Only used if binary is True. Default is 0.1 eV/atom.
			intercept (float): Intercept for linear scaling of stability scores in
				eV/atom. Only used if binary is False. Default is 1.215 eV/atom, which
				is the 99th percentile of the energy above hull values for the MP data
				with theoretical=False.
		"""
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
			response = mpr.request("materials/thermo/?_fields=entries&formula=")
			all_entries = []
			for dct in response:
				all_entries.extend(dct["entries"].values())
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
		if mace_model == "mace-mh-1":
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

	def _ehull(self, xtals: list[Crystal]) -> np.ndarray[float]:
		"""Compute energy above hull for a list of crystals.

		Args:
		    xtals (list[Crystal]): List of crystals to compute energy above hull for.

		Returns:
		   np.ndarray[float]: Array of energy above hull for each crystal.
		"""
		e_above_hulls = np.zeros(len(xtals), dtype=float)
		for idx, xtal in enumerate(xtals):
			try:
				mace_energy = self.calculator.get_potential_energy(xtal.get_ase_atoms())
				gen_entry = ComputedEntry(xtal.get_composition_pymatgen(), mace_energy)
				e_above_hulls[idx] = self.ppd_mp.get_e_above_hull(
					gen_entry, allow_negative=True
				)
			except ValueError:
				e_above_hulls[idx] = np.nan
		return e_above_hulls

	def compute_stability_scores(
		self,
		xtals: list[Crystal],
		e_above_hulls_precomputed: np.ndarray[float] | None = None,
	) -> tuple[np.ndarray[float], np.ndarray[float], float]:
		"""Compute stability scores for a list of crystals.

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
		if self.binary:
			stability_scores[e_above_hulls <= self.threshold] = (
				1.0  # nan <= threshold is False
			)
		else:
			isnan = np.isnan(e_above_hulls)
			stability_scores[~isnan] = np.clip(
				1 - e_above_hulls[~isnan] / self.intercept, 0, 1
			)
		end_time = time.time()
		return stability_scores, e_above_hulls, end_time - start_time
