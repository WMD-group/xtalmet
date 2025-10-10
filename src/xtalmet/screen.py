"""This module offers screening functions based on validity or stability criteria."""

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
from smact.screening import smact_validity

from .crystal import Crystal


def screen_smact(
	xtals: list[Crystal], dir_intermediate: str | None = None
) -> np.ndarray[bool]:
	"""Screen crystals using SMACT.

	Args:
		xtals (list[Crystal]): List of crystals to screen.
		dir_intermediate (str | None): Directory to search for pre-computed screening
			results. If the pre-computed file does not exist in the directory, it will
			be saved to the directory for future use. If set to None, no files will be
			loaded or saved. Default is None.

	Returns:
	    np.ndarray[bool]: Array indicating which crystals pass the screening.

	References:
		- Davies et al., (2019). SMACT: Semiconducting Materials by Analogy and Chemical
		  Theory. Journal of Open Source Software, 4(38), 1361, https://doi.org/10.21105/joss.01361
	"""
	if dir_intermediate is not None:
		path = os.path.join(dir_intermediate, "screen_smact.pkl.gz")
	if dir_intermediate is not None and os.path.exists(path):
		with gzip.open(path, "rb") as f:
			screened = pickle.load(f)
	else:
		screened = np.array(
			[smact_validity(xtal.get_composition_pymatgen()) for xtal in xtals]
		)

	if dir_intermediate is not None and not os.path.exists(path):
		os.makedirs(dir_intermediate, exist_ok=True)
		with gzip.open(path, "wb") as f:
			pickle.dump(screened, f)
	return screened


def screen_ehull(
	xtals: list[Crystal],
	diagram: Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str = "mp_250618",
	dir_intermediate: str | None = None,
) -> np.ndarray[bool]:
	"""Screen crystals using the energy above hull.

	Args:
		xtals (list[Crystal]): List of crystals to screen.
		diagram (Literal["mp_250618", "mp"] | PatchedPhaseDiagram | str): A phased
			diagram to use. If "mp_250618" is specified, the diagram constructed using
			this function from the MP entries on June 18, 2025, will be used. If "mp" is
			specified, the diagram will be constructed on the spot. You can also pass
			your own diagram or a path to it. If the pre-computed screening results
			(screen_ehull.pkl.gz) exist in dir_intermediate, this argument will be
			ignored.
		dir_intermediate (str | None): Directory to search for pre-computed screening
			results. If the pre-computed file does not exist in the directory, it will
			be saved to the directory for future use. If set to None, no files will be
			loaded or saved. Default is None.

	Returns:
		np.ndarray[bool]: Array indicating which crystals pass the screening.
	"""
	if dir_intermediate is not None:
		path_result = os.path.join(dir_intermediate, "screen_ehull.pkl.gz")
	if dir_intermediate is not None and os.path.exists(path_result):
		with gzip.open(path_result, "rb") as f:
			screened = pickle.load(f)
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
		calculator = mace_mp(model="medium-mpa-0")
		screened = np.zeros(len(xtals), dtype=bool)
		e_above_hulls = np.zeros(len(xtals), dtype=float)
		for idx, xtal in enumerate(xtals):
			try:
				mace_energy = calculator.get_potential_energy(xtal.get_ase_atoms())
				gen_entry = ComputedEntry(xtal.get_composition_pymatgen(), mace_energy)
				e_above_hulls[idx] = ppd_mp.get_e_above_hull(
					gen_entry, allow_negative=True
				)
				if e_above_hulls[idx] <= 0.1:
					screened[idx] = True
			except ValueError:
				screened[idx] = False
				e_above_hulls[idx] = np.nan

	if dir_intermediate is not None and not os.path.exists(path_result):
		os.makedirs(dir_intermediate, exist_ok=True)
		with gzip.open(
			os.path.join(dir_intermediate, "screen_ehull.pkl.gz"), "wb"
		) as f:
			pickle.dump(screened, f)
		with gzip.open(os.path.join(dir_intermediate, "ehull.pkl.gz"), "wb") as f:
			pickle.dump(e_above_hulls, f)
	return screened
