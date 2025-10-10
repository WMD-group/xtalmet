"""Generate data files for pytests."""

import gzip
import io
import json
import os
import pickle

import pandas as pd
import requests
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

from xtalmet.crystal import Crystal


def prepare_generated_crystals():
	"""Prepare a set of generated crystal structures for testing."""
	url = "https://raw.githubusercontent.com/hspark1212/chemeleon-dng/main/benchmarks/chemeleon_dng_mp_20_v0.0.2.json.gz"
	response = requests.get(url)
	gen_xtals_all = json.loads(
		gzip.decompress(response.content).decode("utf-8")
	)  # list of dicts
	gen_xtals = gen_xtals_all[:10]  # take only first 10 for testing
	gen_xtals = [
		Crystal.from_Structure(Structure.from_dict(xtal)) for xtal in gen_xtals
	]
	os.makedirs("tests/data", exist_ok=True)
	with gzip.open("tests/data/gen_xtals.pkl.gz", "wb") as f:
		pickle.dump(gen_xtals, f)


def prepare_train_crystals():
	"""Prepare a set of training crystal structures for testing."""
	url = "https://raw.githubusercontent.com/txie-93/cdvae/refs/heads/main/data/mp_20/train.csv"
	response = requests.get(url)
	train_xtals_raw = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
	train_xtals = []
	for idx, row in train_xtals_raw.iterrows():
		structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
		train_xtals.append(Crystal.from_Structure(structure))
		if idx == 9:
			break
	os.makedirs("tests/data", exist_ok=True)
	with gzip.open("tests/data/train_xtals.pkl.gz", "wb") as f:
		pickle.dump(train_xtals, f)


if __name__ == "__main__":
	prepare_generated_crystals()
	prepare_train_crystals()
