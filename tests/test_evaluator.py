"""Test the Evaluator class."""

import datetime
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
		"distance, screen, multiprocessing, n_processes, kwargs",
		[
			("smat", None, False, None, {}),
			("smat", None, True, N_PROCESSES, {}),
			("comp", None, False, None, {}),
			("comp", None, True, None, {}),
			("wyckoff", None, False, None, {}),
			("wyckoff", None, True, N_PROCESSES, {}),
			("magpie", None, False, None, {}),
			("magpie", None, True, None, {}),
			("pdd", None, False, None, {}),
			("pdd", None, True, N_PROCESSES, {}),
			("amd", None, False, None, {}),
			("amd", None, True, None, {}),
			(
				"smat",
				None,
				False,
				None,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				"smat",
				None,
				True,
				N_PROCESSES,
				{"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				"pdd",
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
				True,
				N_PROCESSES,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			("smat", "smact", False, None, {}),
			("comp", "smact", True, None, {}),
			("wyckoff", "smact", False, None, {}),
			(
				"magpie",
				"ehull",
				True,
				N_PROCESSES,
				{"args_screen": {"diagram": "mp_250618"}},
			),
			("pdd", "ehull", False, None, {"args_screen": {"diagram": "mp_250618"}}),
			("amd", "ehull", True, None, {"args_screen": {"diagram": "mp_250618"}}),
		],
	)
	def test_uniqueness(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		distance: str,
		screen: str | None,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test uniqueness."""
		gen_xtals = prepare_gen_xtals
		evaluator = Evaluator(gen_xtals)
		uniqueness_1 = evaluator.uniqueness(
			distance, screen, None, multiprocessing, n_processes, False, **kwargs
		)
		uniqueness_2, times = evaluator.uniqueness(
			distance, screen, tmpdir, multiprocessing, n_processes, True, **kwargs
		)
		assert os.path.exists(os.path.join(tmpdir, f"gen_{distance}.pkl.gz"))
		assert os.path.exists(os.path.join(tmpdir, f"mtx_uni_{distance}.pkl.gz"))
		if screen == "smact":
			assert os.path.exists(os.path.join(tmpdir, "screen_smact.pkl.gz"))
		elif screen == "ehull":
			assert os.path.exists(os.path.join(tmpdir, "screen_ehull.pkl.gz"))
			assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
			if kwargs["args_screen"]["diagram"] == "mp":
				now = datetime.datetime.now()
				year = str(now.year)[-2:]
				month = f"{now.month:02d}"
				day = f"{now.day:02d}"
				assert os.path.exists(
					os.path.join(
						tmpdir,
						f"ppd-mp_all_entries_uncorrected_{year}{month}{day}.pkl.gz",
					)
				)
		assert uniqueness_1 == uniqueness_2
		for key in ["uni_emb", "uni_d_mtx", "uni_metric", "uni_total"]:
			assert key in times
			assert times[key] >= 0

	@pytest.mark.parametrize(
		"download, distance, screen, multiprocessing, n_processes, kwargs",
		[
			(False, "smat", None, False, None, {}),
			(False, "smat", None, True, N_PROCESSES, {}),
			(False, "comp", None, False, None, {}),
			(False, "comp", None, True, None, {}),
			(False, "wyckoff", None, False, None, {}),
			(False, "wyckoff", None, True, N_PROCESSES, {}),
			(False, "magpie", None, False, None, {}),
			(False, "magpie", None, True, None, {}),
			(False, "pdd", None, False, None, {}),
			(False, "pdd", None, True, N_PROCESSES, {}),
			(False, "amd", None, False, None, {}),
			(False, "amd", None, True, None, {}),
			(
				False,
				"smat",
				None,
				False,
				None,
				{"args_mtx": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				False,
				"smat",
				None,
				True,
				N_PROCESSES,
				{"args_mtx": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				False,
				"pdd",
				None,
				False,
				None,
				{
					"args_emb": {"k": 200, "return_row_data": True},
					"args_mtx": {
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
				True,
				None,
				{
					"args_emb": {"k": 100},
					"args_mtx": {
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
				False,
				None,
				{
					"args_emb": {"k": 200},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				False,
				"amd",
				None,
				True,
				N_PROCESSES,
				{
					"args_emb": {"k": 100},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(True, "smat", None, False, None, {}),
			(True, "comp", None, True, None, {}),
			(True, "wyckoff", None, False, None, {}),
			(True, "magpie", None, True, N_PROCESSES, {}),
			(True, "pdd", None, False, None, {}),
			(True, "amd", None, True, None, {}),
			(
				False,
				"smat",
				"ehull",
				False,
				None,
				{"args_screen": {"diagram": "mp_250618"}},
			),
			(
				False,
				"comp",
				"ehull",
				True,
				N_PROCESSES,
				{"args_screen": {"diagram": "mp_250618"}},
			),
			(
				False,
				"wyckoff",
				"ehull",
				False,
				None,
				{"args_screen": {"diagram": "mp_250618"}},
			),
			(False, "magpie", "smact", True, None, {}),
			(False, "pdd", "smact", False, None, {}),
			(False, "amd", "smact", True, N_PROCESSES, {}),
		],
	)
	def test_novelty(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		download: bool,
		distance: str,
		screen: str | None,
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
			screen,
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
			screen,
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
		if screen == "smact":
			assert os.path.exists(os.path.join(tmpdir, "screen_smact.pkl.gz"))
		elif screen == "ehull":
			assert os.path.exists(os.path.join(tmpdir, "screen_ehull.pkl.gz"))
			assert os.path.exists(os.path.join(tmpdir, "ehull.pkl.gz"))
			if kwargs["args_screen"]["diagram"] == "mp":
				now = datetime.datetime.now()
				year = str(now.year)[-2:]
				month = f"{now.month:02d}"
				day = f"{now.day:02d}"
				assert os.path.exists(
					os.path.join(
						tmpdir,
						f"ppd-mp_all_entries_uncorrected_{year}{month}{day}.pkl.gz",
					)
				)
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
