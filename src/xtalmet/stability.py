"""This module offers functions to compute the stability of crystal structures."""

import datetime
import gzip
import os
import pickle
import warnings
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


def compute_ehull(
	xtals: list[Crystal],
	diagram: Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str = "mp_250618",
	dir_intermediate: str | None = None,
) -> np.ndarray[float]:
	"""Compute energy above hull for a list of crystals.

	Args:
		xtals (list[Crystal]): List of crystals to compute energy above hull for.
		diagram (Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str): A phased
			diagram to use. If "mp_250618" is specified, the diagram constructed using
			this function from the MP entries on June 18, 2025, will be used. If "mp" is
			specified, the diagram will be constructed on the spot. You can also pass
			your own diagram or a path to it. If the pre-computed results (ehull.pkl.gz)
			exist in dir_intermediate, this argument will be ignored.
		dir_intermediate (str | None): Directory to search for pre-computed results. If
			the pre-computed file does not exist in the directory, it will be saved to
			the directory for future use. If set to None, no files will be loaded or
			saved. Default is None.

	Returns:
		np.ndarray[float]: Array of energy above hull for each crystal.
	"""
	if dir_intermediate is not None:
		path_result = os.path.join(dir_intermediate, "ehull.pkl.gz")
	if dir_intermediate is not None and os.path.exists(path_result):
		with gzip.open(path_result, "rb") as f:
			e_above_hulls = pickle.load(f)
	else:
		# load or construct a phase diagram
		if isinstance(diagram, PatchedPhaseDiagram):
			ppd_mp = diagram
		elif diagram not in ["mp_250618", "mp"]:
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
			if dir_intermediate is not None:
				os.makedirs(dir_intermediate, exist_ok=True)
				now = datetime.datetime.now()
				year = str(now.year)[-2:]
				month = f"{now.month:02d}"
				day = f"{now.day:02d}"
				with gzip.open(
					os.path.join(
						dir_intermediate,
						f"ppd-mp_all_entries_uncorrected_{year}{month}{day}.pkl.gz",
					),
					"wb",
				) as f:
					pickle.dump(ppd_mp, f)
		# compute energy above hull for each generated crystal
		calculator = mace_mp(model="medium-mpa-0", default_dtype="float64")
		e_above_hulls = np.zeros(len(xtals), dtype=float)
		for idx, xtal in enumerate(xtals):
			try:
				mace_energy = calculator.get_potential_energy(xtal.get_ase_atoms())
				gen_entry = ComputedEntry(xtal.get_composition_pymatgen(), mace_energy)
				e_above_hulls[idx] = ppd_mp.get_e_above_hull(
					gen_entry, allow_negative=True
				)
			except ValueError:
				e_above_hulls[idx] = np.nan

	if dir_intermediate is not None and not os.path.exists(path_result):
		os.makedirs(dir_intermediate, exist_ok=True)
		with gzip.open(os.path.join(dir_intermediate, "ehull.pkl.gz"), "wb") as f:
			pickle.dump(e_above_hulls, f)
	return e_above_hulls


def compute_stability_scores(
	xtals: list[Crystal],
	diagram: Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str = "mp_250618",
	dir_intermediate: str | None = None,
	binary=True,
	**kwargs,
) -> np.ndarray[float]:
	"""Compute stability scores for a list of crystals.

	Args:
		xtals (list[Crystal]): List of crystals to compute stability scores for.
		diagram (Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str): A phased
			diagram to use. If "mp_250618" is specified, the diagram constructed using
			this function from the MP entries on June 18, 2025, will be used. If "mp" is
			specified, the diagram will be constructed on the spot. You can also pass
			your own diagram or a path to it. If the pre-computed results (ehull.pkl.gz)
			exist in dir_intermediate, this argument will be ignored.
		dir_intermediate (str | None): Directory to search for pre-computed results. If
			the pre-computed file does not exist in the directory, it will be saved to
			the directory for future use. If set to None, no files will be loaded or
			saved. Default is None.
		binary (bool): If True, return binary stability scores (1 for stable, 0 for
			unstable). If False, return continuous stability scores between 0 and 1.
			Default is True.
		**kwargs: Additional arguments for stability score computation. If binary is
			True, you can pass "threshold" (float) to set the energy above hull
			threshold for stability (default is 0.1 eV/atom). If binary is False, you
			can pass "intercept" (float) to set the intercept for linear scaling
			(default is 1.215 eV/atom, which is the 99th percentile of the energy above
			hull values for the MP data with theoretical=False).

	Returns:
		np.ndarray[float]: Array of stability scores for each crystal.
	"""
	e_above_hulls = compute_ehull(
		xtals, diagram=diagram, dir_intermediate=dir_intermediate
	)
	stability_scores = np.zeros(len(xtals), dtype=float)
	if binary:
		threshold = kwargs.get("threshold", 0.1)
		stability_scores[e_above_hulls <= threshold] = 1.0  # nan <= threshold is False
	else:
		intercept = kwargs.get("intercept", 1.215)
		stability_scores = np.zeros(len(xtals), dtype=float)
		isnan = np.isnan(e_above_hulls)
		stability_scores[~isnan] = np.clip(1 - e_above_hulls[~isnan] / intercept, 0, 1)
	return stability_scores
