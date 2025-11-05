"""This module offers validity functions to screen crystal structures."""

import gzip
import os
import pickle

import numpy as np
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
		path = os.path.join(dir_intermediate, "valid_smact.pkl.gz")
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
