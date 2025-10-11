"""Convert samples from various models to a list of Crystal objects."""

import gzip
import json
import os
import pickle
import zipfile
from typing import Literal

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

from xtalmet.crystal import Crystal


def convert_cdvae_diffcsppp_format(
	path: str, format: Literal["cdvae", "diffcsppp"]
) -> list[Crystal]:
	"""Convert samples from CDVAE or DiffCSP++ format to a list of Crystal objects.

	Args:
		path (str): Path to the .pt file containing the samples.
		format (Literal): The format of the file, either "cdvae" or "diffcsppp".

	Returns:
		list[Crystal]: A list of Crystal objects.

	Note:
		CDVAE and DiffCSP++ formats are very similar, so we use the same function.
	"""
	gen_xtals_raw = torch.load(path, weights_only=False)

	if format == "cdvae":
		all_frac_coords = gen_xtals_raw["frac_coords"][0]  # (*, 3)
		all_num_atoms = gen_xtals_raw["num_atoms"][0]  # (n_samples, )
		all_atom_types = gen_xtals_raw["atom_types"][0]  # (*, )
		all_lengths = gen_xtals_raw["lengths"][0]  # (n_samples, 3)
		all_angles = gen_xtals_raw["angles"][0]  # (n_samples, 3)
	else:
		all_frac_coords = gen_xtals_raw["frac_coords"]  # (*, 3)
		all_num_atoms = gen_xtals_raw["num_atoms"]  # (n_samples, )
		all_atom_types = gen_xtals_raw["atom_types"]  # (*, )
		all_lengths = gen_xtals_raw["lengths"]  # (n_samples, 3)
		all_angles = gen_xtals_raw["angles"]  # (n_samples, 3)

	start_idx = 0
	gen_xtals = []
	for batch_idx, num_atom in enumerate(all_num_atoms.tolist()):
		gen_xtals.append(
			Crystal(
				lattice=Lattice.from_dict(
					{
						"a": all_lengths[batch_idx][0].item(),
						"b": all_lengths[batch_idx][1].item(),
						"c": all_lengths[batch_idx][2].item(),
						"alpha": all_angles[batch_idx][0].item(),
						"beta": all_angles[batch_idx][1].item(),
						"gamma": all_angles[batch_idx][2].item(),
					}
				),
				species=all_atom_types.narrow(0, start_idx, num_atom).tolist(),
				coords=all_frac_coords.narrow(0, start_idx, num_atom).tolist(),
			)
		)
		start_idx = start_idx + num_atom
	return gen_xtals


def convert_diffcsp_format(path: str) -> list[Crystal]:
	"""Convert samples from DiffCSP format to a list of Crystal objects.

	Args:
		path (str): Path to the .json file containing the samples.

	Returns:
		list[Crystal]: A list of Crystal objects.
	"""
	with open(path) as f:
		gen_xtals_raw = json.load(f)
	gen_xtals = [
		Crystal.from_Structure(Structure.from_dict(xtal)) for xtal in gen_xtals_raw
	]
	return gen_xtals


def convert_mattergen_format(path: str) -> list[Crystal]:
	"""Convert samples from MatterGen format to a list of Crystal objects.

	Args:
		path (str): Path to the .zip file containing the samples.

	Returns:
		list[Crystal]: A list of Crystal objects.
	"""
	gen_xtals_raw = []
	with zipfile.ZipFile(path, "r") as zf:
		all_files = zf.namelist()
		for filename in all_files:
			if filename.endswith(".cif"):
				with zf.open(filename) as f:
					content = f.read().decode("utf-8")
					gen_xtals_raw.append(content)
	gen_xtals = []
	for xtal in gen_xtals_raw:
		structure = CifParser.from_str(xtal).parse_structures(primitive=True)[0]
		gen_xtals.append(Crystal.from_Structure(structure))
	return gen_xtals


def convert():
	"""Convert samples from various models to a list of Crystal objects."""
	paths_raw = {
		"adit": os.path.join(
			os.path.dirname(__file__), "raw/mp20/adit/adit_dng_mp_20.json"
		),
		"cdvae": os.path.join(os.path.dirname(__file__), "raw/mp20/cdvae/eval_gen.pt"),
		"chemeleon": os.path.join(
			os.path.dirname(__file__), "raw/mp20/chemeleon/generated_structures.json"
		),
		"diffcsp": os.path.join(
			os.path.dirname(__file__), "raw/mp20/diffcsp/diffcsp_dng_mp_20.json"
		),
		"diffcsppp": os.path.join(
			os.path.dirname(__file__), "raw/mp20/diffcsppp/eval_gen.pt"
		),
		"mattergen": os.path.join(
			os.path.dirname(__file__), "raw/mp20/mattergen/generated_crystals_cif.zip"
		),
	}

	for model in ["adit", "cdvae", "chemeleon", "diffcsp", "diffcsppp", "mattergen"]:
		if model in ["cdvae", "diffcsppp"]:
			gen_xtals = convert_cdvae_diffcsppp_format(paths_raw[model], model)
		elif model in ["diffcsp", "adit", "chemeleon"]:
			gen_xtals = convert_diffcsp_format(paths_raw[model])
		else:  # mattergen
			gen_xtals = convert_mattergen_format(paths_raw[model])
		path_processed = os.path.join(
			os.path.dirname(__file__), f"hf/mp20/model/{model}.pkl.gz"
		)
		os.makedirs(os.path.dirname(path_processed), exist_ok=True)
		with gzip.open(path_processed, "wb") as f:
			pickle.dump(gen_xtals, f)


if __name__ == "__main__":
	convert()
