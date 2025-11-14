"""Computing the embeddings for datasets."""

import gzip
import io
import json
import os
import pickle
import time

import pandas as pd
import requests
from pymatgen.io.cif import CifParser

from xtalmet.constants import SUPPORTED_DISTANCES
from xtalmet.crystal import Crystal
from xtalmet.distance import _compute_embeddings


def dataset_embeddings():
	"""Computing the embeddings for datasets."""
	url = "https://raw.githubusercontent.com/txie-93/cdvae/refs/heads/main/data/mp_20/train.csv"
	response = requests.get(url)
	train_xtals_raw = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
	train_xtals = []
	for _, row in train_xtals_raw.iterrows():
		structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
		train_xtals.append(Crystal.from_Structure(structure))

	dir_save = os.path.join(os.path.dirname(__file__), "hf", "mp20", "train")
	os.makedirs(dir_save, exist_ok=True)
	times = {}
	for distance in SUPPORTED_DISTANCES:
		start = time.time()
		embs = _compute_embeddings(
			distance, train_xtals, multiprocessing=True, n_processes=10
		)
		end = time.time()
		times[distance] = end - start
		with gzip.open(os.path.join(dir_save, f"train_{distance}.pkl.gz"), "wb") as f:
			pickle.dump(embs, f)

	with open(os.path.join(dir_save, "times.json"), "w") as f:
		json.dump(times, f, indent=4, sort_keys=True)


if __name__ == "__main__":
	dataset_embeddings()
