"""Test the Evaluator class."""

import gzip
import os
import pickle
from multiprocessing import cpu_count
from typing import Literal

import pytest
from pymatgen.core.structure import Structure

from xtalmet.crystal import Crystal
from xtalmet.evaluator import Evaluator

N_PROCESSES = max(cpu_count() // 2 - 1, 1)


@pytest.fixture(scope="module")
def prepare_gen_xtals() -> list[Crystal]:
	"""Prepare a set of generated crystal structures for testing."""
	with gzip.open("tests/data/gen_xtals.pkl.gz", "rb") as f:
		gen_xtals = pickle.load(f)
	return gen_xtals


@pytest.fixture(scope="module")
def prepare_train_xtals() -> list[Crystal]:
	"""Prepare a set of training crystal structures for testing."""
	with gzip.open("tests/data/train_xtals.pkl.gz", "rb") as f:
		train_xtals = pickle.load(f)
	return train_xtals


class TestEvaluator:
	"""Test the Evaluator class."""

	@pytest.mark.parametrize(
		"validity, stability, uniqueness, novelty, distance, ref_xtals, agg_func, weights, multiprocessing, n_processes, kwargs",
		[
			# only validity
			(["smact"], None, False, False, None, None, "prod", None, False, None, {}),
			(
				["structure"],
				None,
				False,
				False,
				None,
				None,
				"ave",
				None,
				False,
				None,
				{},
			),
			(
				["smact", "structure"],
				None,
				False,
				False,
				None,
				None,
				"ave",
				{"validity": 1.0},
				False,
				None,
				{
					"args_validity": {
						"structure": {
							"threshold_distance": 0.5,
							"threshold_volume": 0.1,
						}
					}
				},
			),
			# only stability
			(None, "binary", False, False, None, None, "prod", None, False, None, {}),
			(
				None,
				"continuous",
				False,
				False,
				None,
				None,
				"ave",
				None,
				False,
				None,
				{},
			),
			(
				None,
				"binary",
				False,
				False,
				None,
				None,
				"ave",
				{"stability": 1.0},
				False,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
						"threshold": 0.1,
					}
				},
			),
			(
				None,
				"continuous",
				False,
				False,
				None,
				None,
				"prod",
				None,
				False,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "mh-1",
						"intercept": 0.4289,
					}
				},
			),
			# only uniqueness
			(None, None, True, False, "smat", None, "prod", None, False, None, {}),
			(
				None,
				None,
				True,
				False,
				"smat",
				None,
				"ave",
				None,
				False,
				None,
				{"args_dist": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}},
			),
			(None, None, True, False, "comp", None, "prod", None, False, None, {}),
			(
				None,
				None,
				True,
				False,
				"wyckoff",
				None,
				"ave",
				{"uniqueness": 1.0},
				False,
				None,
				{},
			),
			(None, None, True, False, "magpie", None, "prod", None, False, None, {}),
			(None, None, True, False, "pdd", None, "ave", None, False, None, {}),
			(
				None,
				None,
				True,
				False,
				"pdd",
				None,
				"prod",
				None,
				False,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": False,
					},
				},
			),
			(
				None,
				None,
				True,
				False,
				"amd",
				None,
				"ave",
				{"uniqueness": 1.0},
				False,
				None,
				{},
			),
			(
				None,
				None,
				True,
				False,
				"amd",
				None,
				"prod",
				None,
				False,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(None, None, True, False, "elmd", None, "ave", None, False, None, {}),
			(
				None,
				None,
				True,
				False,
				"elmd",
				None,
				"prod",
				None,
				False,
				None,
				{"args_dist": {"metric": "mod_petti"}},
			),
			# only novelty
			(None, None, False, True, "smat", "mp20", "prod", None, False, None, {}),
			(
				None,
				None,
				False,
				True,
				"smat",
				None,
				"ave",
				None,
				True,
				None,
				{"args_dist": {"ltol": 0.2, "stol": 0.3, "angle_tol": 5}},
			),
			(None, None, False, True, "comp", "mp20", "prod", None, False, None, {}),
			(
				None,
				None,
				False,
				True,
				"wyckoff",
				None,
				"ave",
				{"novelty": 1.0},
				True,
				N_PROCESSES,
				{},
			),
			(None, None, False, True, "magpie", "mp20", "prod", None, False, None, {}),
			(None, None, False, True, "pdd", None, "ave", None, True, None, {}),
			(
				None,
				None,
				False,
				True,
				"pdd",
				"mp20",
				"prod",
				None,
				False,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": False,
					},
				},
			),
			(
				None,
				None,
				False,
				True,
				"amd",
				None,
				"ave",
				{"novelty": 1.0},
				True,
				N_PROCESSES,
				{},
			),
			(
				None,
				None,
				False,
				True,
				"amd",
				"mp20",
				"prod",
				None,
				False,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(None, None, False, True, "elmd", None, "ave", None, True, None, {}),
			(
				None,
				None,
				False,
				True,
				"elmd",
				"mp20",
				"prod",
				None,
				False,
				None,
				{"args_dist": {"metric": "mod_petti"}},
			),
			# vsun
			(
				["smact"],
				"binary",
				True,
				True,
				"smat",
				"mp20",
				"prod",
				None,
				False,
				None,
				{},
			),
			(
				["structure"],
				"continuous",
				True,
				True,
				"comp",
				None,
				"ave",
				None,
				True,
				None,
				{},
			),
			(
				["smact", "structure"],
				"binary",
				True,
				True,
				"wyckoff",
				"mp20",
				"ave",
				{
					"validity": 0.25,
					"stability": 0.25,
					"uniqueness": 0.25,
					"novelty": 0.25,
				},
				False,
				None,
				{},
			),
			(
				["smact"],
				"continuous",
				True,
				True,
				"magpie",
				None,
				"prod",
				None,
				True,
				N_PROCESSES,
				{},
			),
			(
				["structure"],
				"binary",
				True,
				True,
				"pdd",
				"mp20",
				"ave",
				None,
				False,
				None,
				{},
			),
			(
				["smact", "structure"],
				"continuous",
				True,
				True,
				"amd",
				None,
				"ave",
				{
					"validity": 0.2,
					"stability": 0.3,
					"uniqueness": 0.2,
					"novelty": 0.3,
				},
				True,
				None,
				{},
			),
			(
				["smact"],
				"binary",
				True,
				True,
				"elmd",
				"mp20",
				"prod",
				None,
				False,
				None,
				{},
			),
			(
				["smact", "structure"],
				"continuous",
				True,
				True,
				"amd",
				None,
				"ave",
				None,
				True,
				N_PROCESSES,
				{
					"args_validity": {
						"structure": {
							"threshold_distance": 0.5,
							"threshold_volume": 0.1,
						}
					},
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "mh-1",
						"intercept": 0.4289,
					},
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
		],
	)
	def test_init_evaluate(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		validity: list[str] | None,
		stability: str | None,
		uniqueness: bool,
		novelty: bool,
		distance: str | None,
		ref_xtals: list[Crystal | Structure] | str | None,
		agg_func: Literal["prod", "ave"],
		weights: dict[str, float] | None,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test __init__ and evaluate."""
		if novelty and ref_xtals is None:
			ref_xtals = prepare_train_xtals
		evaluator = Evaluator(
			validity=validity,
			stability=stability,
			uniqueness=uniqueness,
			novelty=novelty,
			distance=distance,
			ref_xtals=ref_xtals,
			agg_func=agg_func,
			weights=weights,
			multiprocessing=multiprocessing,
			n_processes=n_processes,
			**kwargs,
		)
		vsun_1, scores_1, times_1 = evaluator.evaluate(
			xtals=prepare_gen_xtals,
			dir_intermediate=tmpdir,
			multiprocessing=multiprocessing,
			n_processes=n_processes,
		)
		vsun_2, scores_2, times_2 = evaluator.evaluate(
			xtals=prepare_gen_xtals,
			dir_intermediate=tmpdir,
			multiprocessing=multiprocessing,
			n_processes=n_processes,
		)
		times_key = set()
		if validity is not None and "smact" in validity:
			for method in validity:
				assert os.path.exists(os.path.join(tmpdir, f"val_{method}.pkl.gz"))
				times_key.add(f"val_{method}")
		if stability is not None:
			assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
			times_key.add("stab")
		if uniqueness:
			assert os.path.exists(os.path.join(tmpdir, f"emb_{distance}.pkl.gz"))
			assert os.path.exists(os.path.join(tmpdir, f"mtx_uni_{distance}.pkl.gz"))
			times_key.add("uni_emb")
			times_key.add("uni_d_mtx")
		if novelty:
			assert os.path.exists(os.path.join(tmpdir, f"emb_{distance}.pkl.gz"))
			assert os.path.exists(os.path.join(tmpdir, f"mtx_nov_{distance}.pkl.gz"))
			times_key.add("nov_emb")
			times_key.add("nov_d_mtx")
		times_key.add("aggregation")
		times_key.add("total")
		assert vsun_1 == vsun_2
		assert all(scores_1 == scores_2)
		for times in [times_1, times_2]:
			for key in times_key:
				assert key in times
				assert times[key] >= 0
