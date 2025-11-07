"""Test the Evaluator class."""

import gzip
import os
import pickle
from multiprocessing import cpu_count

import pytest

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

	def test_init(self, prepare_gen_xtals: list[Crystal]):
		"""Test __init__."""
		gen_xtals = prepare_gen_xtals
		_ = Evaluator(gen_xtals)
		assert True

	@pytest.mark.parametrize(
		"distance, validity, stability, multiprocessing, n_processes, kwargs",
		[
			("smat", None, None, False, None, {}),
			("smat", None, None, True, N_PROCESSES, {}),
			("comp", None, None, False, None, {}),
			("comp", None, None, True, None, {}),
			("wyckoff", None, None, False, None, {}),
			("wyckoff", None, None, True, N_PROCESSES, {}),
			("magpie", None, None, False, None, {}),
			("magpie", None, None, True, None, {}),
			("pdd", None, None, False, None, {}),
			("pdd", None, None, True, N_PROCESSES, {}),
			("amd", None, None, False, None, {}),
			("amd", None, None, True, None, {}),
			(
				"smat",
				None,
				None,
				False,
				None,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				"smat",
				None,
				None,
				True,
				N_PROCESSES,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				"pdd",
				None,
				None,
				False,
				None,
				{
					"args_emb": {"k": 200, "return_row_data": True},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": True,
					},
				},
			),
			(
				"pdd",
				None,
				None,
				True,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": None,
						"verbose": False,
					},
				},
			),
			(
				"amd",
				None,
				None,
				False,
				None,
				{
					"args_emb": {"k": 200},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				"amd",
				None,
				None,
				True,
				N_PROCESSES,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			("smat", None, "binary", False, None, {}),
			("comp", None, "continuous", True, None, {}),
			("wyckoff", None, "binary", False, None, {}),
			(
				"magpie",
				None,
				"binary",
				True,
				N_PROCESSES,
				{"args_stability": {"diagram": "mp_250618", "mace_model": "mace-mh-1"}},
			),
			(
				"pdd",
				None,
				"continuous",
				False,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
					}
				},
			),
			(
				"amd",
				None,
				"binary",
				True,
				None,
				{"args_stability": {"diagram": "mp_250618", "mace_model": "mace-mh-1"}},
			),
			("smat", ["smact"], None, False, None, {}),
			("comp", ["smact"], "binary", True, None, {}),
			("wyckoff", ["smact"], "continuous", False, None, {}),
			(
				"magpie",
				["smact"],
				None,
				True,
				N_PROCESSES,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
					}
				},
			),
			(
				"pdd",
				["smact"],
				"binary",
				False,
				None,
				{"args_stability": {"diagram": "mp_250618", "mace_model": "mace-mh-1"}},
			),
			(
				"amd",
				["smact"],
				"continuous",
				True,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
					}
				},
			),
		],
	)
	def test_uniqueness(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		distance: str,
		validity: list[str] | None,
		stability: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test uniqueness."""
		gen_xtals = prepare_gen_xtals
		evaluator = Evaluator(gen_xtals)
		uniqueness_1 = evaluator.uniqueness(
			distance,
			validity,
			stability,
			None,
			multiprocessing,
			n_processes,
			False,
			**kwargs,
		)
		uniqueness_2, times = evaluator.uniqueness(
			distance,
			validity,
			stability,
			tmpdir,
			multiprocessing,
			n_processes,
			True,
			**kwargs,
		)
		assert os.path.exists(os.path.join(tmpdir, f"gen_{distance}.pkl.gz"))
		assert os.path.exists(os.path.join(tmpdir, f"mtx_uni_{distance}.pkl.gz"))
		if validity is not None and "smact" in validity:
			assert os.path.exists(os.path.join(tmpdir, "valid_smact.pkl.gz"))
		if stability is not None:
			assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
		assert uniqueness_1 == uniqueness_2
		for key in ["uni_emb", "uni_d_mtx", "uni_metric", "uni_total"]:
			assert key in times
			assert times[key] >= 0

	@pytest.mark.parametrize(
		"download, distance, validity, stability, multiprocessing, n_processes, kwargs",
		[
			(False, "smat", None, None, False, None, {}),
			(False, "smat", None, None, True, N_PROCESSES, {}),
			(False, "comp", None, None, False, None, {}),
			(False, "comp", None, None, True, None, {}),
			(False, "wyckoff", None, None, False, None, {}),
			(False, "wyckoff", None, None, True, N_PROCESSES, {}),
			(False, "magpie", None, None, False, None, {}),
			(False, "magpie", None, None, True, None, {}),
			(False, "pdd", None, None, False, None, {}),
			(False, "pdd", None, None, True, N_PROCESSES, {}),
			(False, "amd", None, None, False, None, {}),
			(False, "amd", None, None, True, None, {}),
			(
				False,
				"smat",
				None,
				None,
				False,
				None,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				False,
				"smat",
				None,
				None,
				True,
				N_PROCESSES,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				False,
				"pdd",
				None,
				None,
				False,
				None,
				{
					"args_emb": {"k": 200, "return_row_data": True},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": True,
					},
				},
			),
			(
				False,
				"pdd",
				None,
				None,
				True,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": None,
						"verbose": False,
					},
				},
			),
			(
				False,
				"amd",
				None,
				None,
				False,
				None,
				{
					"args_emb": {"k": 200},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				False,
				"amd",
				None,
				None,
				True,
				N_PROCESSES,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(True, "smat", None, "continuous", False, None, {}),
			(True, "comp", None, "binary", True, None, {}),
			(True, "wyckoff", None, "continuous", False, None, {}),
			(
				True,
				"magpie",
				None,
				"continuous",
				True,
				N_PROCESSES,
				{"args_stability": {"diagram": "mp_250618", "mace_model": "mace-mh-1"}},
			),
			(
				True,
				"pdd",
				None,
				"binary",
				False,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
					}
				},
			),
			(
				True,
				"amd",
				None,
				"continuous",
				True,
				None,
				{"args_stability": {"diagram": "mp_250618", "mace_model": "mace-mh-1"}},
			),
			(
				False,
				"smat",
				["smact"],
				None,
				False,
				None,
				{},
			),
			(
				False,
				"comp",
				["smact"],
				"binary",
				True,
				N_PROCESSES,
				{},
			),
			(
				False,
				"wyckoff",
				["smact"],
				"continuous",
				False,
				None,
				{
					"args_stability": {
						"diagram": "mp_250618",
						"mace_model": "medium-mpa-0",
					}
				},
			),
			(False, "magpie", ["smact"], None, True, None, {}),
			(False, "pdd", ["smact"], "binary", False, None, {}),
			(False, "amd", ["smact"], "continuous", True, N_PROCESSES, {}),
		],
	)
	def test_novelty(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		download: bool,
		distance: str,
		validity: list[str] | None,
		stability: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test novelty."""
		gen_xtals = prepare_gen_xtals
		train_xtals = "mp20" if download else prepare_train_xtals
		evaluator = Evaluator(gen_xtals)
		novelty_1 = evaluator.novelty(
			train_xtals,
			distance,
			validity,
			stability,
			None,
			None,
			multiprocessing,
			n_processes,
			False,
			**kwargs,
		)
		novelty_2, times = evaluator.novelty(
			train_xtals,
			distance,
			validity,
			stability,
			tmpdir,
			tmpdir,
			multiprocessing,
			n_processes,
			True,
			**kwargs,
		)
		assert os.path.exists(os.path.join(tmpdir, f"gen_{distance}.pkl.gz"))
		if not download:
			assert os.path.exists(os.path.join(tmpdir, f"train_{distance}.pkl.gz"))
		assert os.path.exists(os.path.join(tmpdir, f"mtx_nov_{distance}.pkl.gz"))
		if validity is not None and "smact" in validity:
			assert os.path.exists(os.path.join(tmpdir, "valid_smact.pkl.gz"))
		if stability is not None:
			assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
		assert novelty_1 == novelty_2
		for key in [
			"nov_emb_gen",
			"nov_emb_train",
			"nov_d_mtx",
			"nov_metric",
			"nov_total",
		]:
			assert key in times
			assert times[key] >= 0
