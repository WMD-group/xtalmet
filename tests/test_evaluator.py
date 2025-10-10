"""Test the Evaluator class."""

import datetime
import gzip
import os
import pickle
from typing import Literal

import pytest

from xtalmet.crystal import Crystal
from xtalmet.evaluator import Evaluator


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
		"distance, kwargs",
		[
			("smat", {}),
			("comp", {}),
			("wyckoff", {}),
			("magpie", {}),
			("pdd", {}),
			("amd", {}),
			("pdd", {"k": 200, "return_row_data": True}),
			("amd", {"k": 200}),
		],
	)
	def test_embed(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		kwargs: dict,
	):
		"""Test _embed."""
		gen_xtals = prepare_gen_xtals
		train_xtals = prepare_train_xtals
		evaluator = Evaluator(gen_xtals)
		for datatype, xtals in [("gen", gen_xtals), ("train", train_xtals)]:
			embs_1, _ = evaluator._embed(xtals, distance, None, datatype, **kwargs)
			embs_2, _ = evaluator._embed(xtals, distance, tmpdir, datatype, **kwargs)
			if distance in ["pdd", "amd"]:
				assert all((embs_1[i] == embs_2[i]).all() for i in range(len(embs_1)))
			else:
				assert embs_1 == embs_2
			assert os.path.exists(os.path.join(tmpdir, f"{datatype}_{distance}.pkl.gz"))

	@pytest.mark.parametrize(
		"distance, kwargs",
		[
			("smat", {}),
			("comp", {}),
			("wyckoff", {}),
			("magpie", {}),
			("pdd", {}),
			("amd", {}),
			("smat", {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}),
			(
				"pdd",
				{
					"metric": "chebyshev",
					"backend": "multiprocessing",
					"n_jobs": 2,
					"verbose": True,
				},
			),
			(
				"pdd",
				{
					"metric": "chebyshev",
					"backend": "multiprocessing",
					"n_jobs": None,
					"verbose": False,
				},
			),
			("amd", {"metric": "chebyshev", "low_memory": False}),
		],
	)
	def test_distance_matrix(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		kwargs: dict,
	):
		"""Test _distance_matrix."""
		gen_xtals = prepare_gen_xtals
		train_xtals = prepare_train_xtals
		evaluator = Evaluator(gen_xtals)
		embs_gen, _ = evaluator._embed(gen_xtals, distance, None, "gen")
		embs_train, _ = evaluator._embed(train_xtals, distance, None, "train")
		for met, embs in [("uni", embs_gen), ("nov", embs_train)]:
			d_mtx_1, _ = evaluator._distance_matrix(
				embs_gen, embs, distance, None, met, **kwargs
			)
			d_mtx_2, _ = evaluator._distance_matrix(
				embs_gen, embs, distance, tmpdir, met, **kwargs
			)
			assert (d_mtx_1 == d_mtx_2).all()
			assert os.path.exists(os.path.join(tmpdir, f"mtx_{met}_{distance}.pkl.gz"))

	@pytest.mark.parametrize(
		"distance, screen, kwargs",
		[
			("smat", None, {}),
			("comp", None, {}),
			("wyckoff", None, {}),
			("magpie", None, {}),
			("pdd", None, {}),
			("amd", None, {}),
			("smat", None, {"args_mtx": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}}),
			(
				"pdd",
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
				"pdd",
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
				"amd",
				None,
				{
					"args_emb": {"k": 200},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				"amd",
				None,
				{
					"args_emb": {"k": 100},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			("smat", "smact", {}),
			("comp", "smact", {}),
			("wyckoff", "smact", {}),
			("magpie", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
			("pdd", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
			("amd", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
		],
	)
	def test_uniqueness(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		screen: Literal[None, "smact", "ehull"],
		kwargs: dict,
	):
		"""Test uniqueness."""
		gen_xtals = prepare_gen_xtals
		evaluator = Evaluator(gen_xtals)
		uniqueness_1 = evaluator.uniqueness(distance, screen, None, False, **kwargs)
		uniqueness_2, times = evaluator.uniqueness(
			distance, screen, tmpdir, True, **kwargs
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
			assert times[key] > 0

	@pytest.mark.parametrize(
		"download, distance, screen, kwargs",
		[
			(False, "smat", None, {}),
			(False, "comp", None, {}),
			(False, "wyckoff", None, {}),
			(False, "magpie", None, {}),
			(False, "pdd", None, {}),
			(False, "amd", None, {}),
			(
				False,
				"smat",
				None,
				{"args_mtx": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}},
			),
			(
				False,
				"pdd",
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
				{
					"args_emb": {"k": 200},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				False,
				"amd",
				None,
				{
					"args_emb": {"k": 100},
					"args_mtx": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(True, "smat", None, {}),
			(True, "comp", None, {}),
			(True, "wyckoff", None, {}),
			(True, "magpie", None, {}),
			(True, "pdd", None, {}),
			(True, "amd", None, {}),
			(False, "smat", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
			(False, "comp", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
			(False, "wyckoff", "ehull", {"args_screen": {"diagram": "mp_250618"}}),
			(False, "magpie", "smact", {}),
			(False, "pdd", "smact", {}),
			(False, "amd", "smact", {}),
		],
	)
	def test_novelty(
		self,
		tmpdir: str,
		prepare_gen_xtals: list[Crystal],
		prepare_train_xtals: list[Crystal],
		download: bool,
		distance: Literal["smat", "comp", "wyckoff", "magpie", "pdd", "amd"],
		screen: Literal[None, "smact", "ehull"],
		kwargs: dict,
	):
		"""Test novelty."""
		gen_xtals = prepare_gen_xtals
		train_xtals = "mp20" if download else prepare_train_xtals
		evaluator = Evaluator(gen_xtals)
		novelty_1 = evaluator.novelty(
			train_xtals, distance, screen, None, False, **kwargs
		)
		novelty_2, times = evaluator.novelty(
			train_xtals, distance, screen, tmpdir, True, **kwargs
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
			assert times[key] > 0
