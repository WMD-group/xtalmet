"""Test distance functions."""

import os
from multiprocessing import cpu_count

import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pytest import approx

from xtalmet.constants import BINARY_DISTANCES
from xtalmet.crystal import Crystal
from xtalmet.distance import (
	_compute_embeddings,
	_d_amd,
	_d_comp,
	_d_elmd,
	_d_magpie,
	_d_pdd,
	_d_smat,
	_d_wyckoff,
	_distance_matrix_d_amd,
	_distance_matrix_d_comp,
	_distance_matrix_d_elmd,
	_distance_matrix_d_magpie,
	_distance_matrix_d_pdd,
	_distance_matrix_d_smat,
	_distance_matrix_d_wyckoff,
	distance,
	distance_matrix,
)

N_PROCESSES = max(cpu_count() // 2 - 1, 1)


@pytest.fixture(
	scope="module",
	params=[
		(
			"wz-ZnO",
			"rs-ZnO",
			{
				"smat": 1.0,
				"comp": 0.0,
				"wyckoff": 1.0,
				"magpie": 0.000e00,
				"amd": 1.042e00,
				"pdd": 1.042e00,
				"elmd": 0.0,
			},
		),
		(
			"wz-ZnO",
			"wz-GaN",
			{
				"smat": 1.0,
				"comp": 1.0,
				"wyckoff": 0.0,
				"magpie": 6.298e02,
				"amd": 9.684e-02,
				"pdd": 9.684e-02,
				"elmd": 7.0,
			},
		),
		(
			"wz-ZnO",
			"Bi2Te3",
			{
				"smat": 1.0,
				"comp": 1.0,
				"wyckoff": 1.0,
				"magpie": 1.070e03,
				"amd": 3.240e00,
				"pdd": 3.276e00,
				"elmd": 10.7,
			},
		),
	],
)
def prepare(request) -> tuple[Structure, Structure, Crystal, Crystal, dict]:
	"""Prepare Crystal objects for testing."""
	struc_1 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), f"data/{request.param[0]}.cif")
	)
	struc_2 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), f"data/{request.param[1]}.cif")
	)
	xtal_1 = Crystal.from_Structure(struc_1)
	xtal_2 = Crystal.from_Structure(struc_2)
	return (struc_1, struc_2, xtal_1, xtal_2, request.param[2])


@pytest.fixture(scope="module")
def prepare_four_xtals() -> tuple[Crystal, Crystal, Crystal, Crystal]:
	"""Prepare four Crystal objects for testing distance matrix."""
	xtal_1 = Crystal.from_Structure(
		Structure.from_file(os.path.join(os.path.dirname(__file__), "data/wz-ZnO.cif"))
	)
	xtal_2 = Crystal.from_Structure(
		Structure.from_file(os.path.join(os.path.dirname(__file__), "data/rs-ZnO.cif"))
	)
	xtal_3 = Crystal.from_Structure(
		Structure.from_file(os.path.join(os.path.dirname(__file__), "data/wz-GaN.cif"))
	)
	xtal_4 = Crystal.from_Structure(
		Structure.from_file(os.path.join(os.path.dirname(__file__), "data/Bi2Te3.cif"))
	)
	return (xtal_1, xtal_2, xtal_3, xtal_4)


@pytest.fixture(scope="module")
def prepare_d_mtx() -> tuple[Structure, Structure, Structure, Structure, dict]:
	"""Prepare distance matrices."""
	struc_1 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), "data/wz-ZnO.cif")
	)
	struc_2 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), "data/rs-ZnO.cif")
	)
	struc_3 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), "data/wz-GaN.cif")
	)
	struc_4 = Structure.from_file(
		os.path.join(os.path.dirname(__file__), "data/Bi2Te3.cif")
	)
	xtal_1 = Crystal.from_Structure(struc_1)
	xtal_2 = Crystal.from_Structure(struc_2)
	xtal_3 = Crystal.from_Structure(struc_3)
	xtal_4 = Crystal.from_Structure(struc_4)
	xtals = [xtal_1, xtal_2, xtal_3, xtal_4]
	expected = {}
	for dist, func in {
		"smat": _d_smat,
		"comp": _d_comp,
		"wyckoff": _d_wyckoff,
		"magpie": _d_magpie,
		"pdd": _d_pdd,
		"amd": _d_amd,
		"elmd": _d_elmd,
	}.items():
		embs = [xtal.get_embedding(dist) for xtal in xtals]
		d_mtx = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				d_mtx[i, j] = func(embs[i], embs[j])
		expected[dist] = d_mtx
	return (struc_1, struc_2, struc_3, struc_4, expected)


class TestDistance:
	"""Test distance functions."""

	@pytest.mark.parametrize("kwargs", [{}, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}])
	def test_d_smat(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test _d_smat."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert _d_smat(xtal_1, xtal_2, **kwargs) == expected["smat"]
		matcher = StructureMatcher(**kwargs)
		assert _d_smat(xtal_1, xtal_2, matcher=matcher) == expected["smat"]

	def test_d_comp(self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]):
		"""Test _d_comp."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert (
			_d_comp(xtal_1._get_emb_d_comp(), xtal_2._get_emb_d_comp())
			== expected["comp"]
		)

	def test_d_wyckoff(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]
	):
		"""Test _d_wyckoff."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert (
			_d_wyckoff(xtal_1._get_emb_d_wyckoff(), xtal_2._get_emb_d_wyckoff())
			== expected["wyckoff"]
		)

	def test_d_magpie(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]
	):
		"""Test _d_magpie."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert _d_magpie(
			xtal_1._get_emb_d_magpie(), xtal_2._get_emb_d_magpie()
		) == approx(expected["magpie"], rel=1e-3)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{
				"metric": "chebyshev",
				"backend": "multiprocessing",
				"n_jobs": 2,
				"verbose": True,
			},
			{
				"metric": "chebyshev",
				"backend": "multiprocessing",
				"n_jobs": None,
				"verbose": False,
			},
		],
	)
	def test_d_pdd(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test _d_pdd."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert _d_pdd(
			xtal_1._get_emb_d_pdd(), xtal_2._get_emb_d_pdd(), **kwargs
		) == approx(expected["pdd"], rel=1e-3)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"metric": "chebyshev", "low_memory": False},
		],
	)
	def test_d_amd(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test d_amd."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert _d_amd(
			xtal_1._get_emb_d_amd(), xtal_2._get_emb_d_amd(), **kwargs
		) == approx(expected["amd"], rel=1e-3)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"metric": "mod_petti"},
			{"metric": "fast"},
		],
	)
	def test_d_elmd(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test d_elmd."""
		_, _, xtal_1, xtal_2, expected = prepare
		assert _d_elmd(
			xtal_1._get_emb_d_elmd(), xtal_2._get_emb_d_elmd(), **kwargs
		) == approx(expected["elmd"], rel=1e-3)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes, kwargs",
		[
			(False, None, {}),
			(True, N_PROCESSES, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}),
			(True, None, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}),
		],
	)
	def test_distance_matrix_d_smat(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test _distance_matrix_d_smat."""
		xtals = prepare_four_xtals
		embs = [xtal.get_embedding("smat") for xtal in xtals]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_smat(embs[i], embs[j], **kwargs)
		results_1 = _distance_matrix_d_smat(
			embs, None, multiprocessing, n_processes, **kwargs
		)
		assert np.all(results_1 == expected)
		results_2 = _distance_matrix_d_smat(
			embs, embs, multiprocessing, n_processes, **kwargs
		)
		assert np.all(results_2 == expected)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes",
		[(False, None), (True, N_PROCESSES), (True, None)],
	)
	def test_distance_matrix_d_comp(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		multiprocessing: bool,
		n_processes: int | None,
	):
		"""Test _distance_matrix_d_comp."""
		xtals = prepare_four_xtals
		embs = [xtal.get_embedding("comp") for xtal in xtals]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_comp(embs[i], embs[j])
		results_1 = _distance_matrix_d_comp(embs, None, multiprocessing, n_processes)
		assert np.all(results_1 == expected)
		results_2 = _distance_matrix_d_comp(embs, embs, multiprocessing, n_processes)
		assert np.all(results_2 == expected)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes",
		[(False, None), (True, N_PROCESSES), (True, None)],
	)
	def test_distance_matrix_d_wyckoff(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		multiprocessing: bool,
		n_processes: int | None,
	):
		"""Test _distance_matrix_d_wyckoff."""
		xtals = prepare_four_xtals
		embs = [xtal.get_embedding("wyckoff") for xtal in xtals]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_wyckoff(embs[i], embs[j])
		results_1 = _distance_matrix_d_wyckoff(embs, None, multiprocessing, n_processes)
		assert np.all(results_1 == expected)
		results_2 = _distance_matrix_d_wyckoff(embs, embs, multiprocessing, n_processes)
		assert np.all(results_2 == expected)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes",
		[(False, None), (True, N_PROCESSES), (True, None)],
	)
	def test_distance_matrix_d_magpie(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		multiprocessing: bool,
		n_processes: int | None,
	):
		"""Test _distance_matrix_d_magpie."""
		xtals = prepare_four_xtals
		embs = [xtal.get_embedding("magpie") for xtal in xtals]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_magpie(embs[i], embs[j])
		results_1 = _distance_matrix_d_magpie(embs, None, multiprocessing, n_processes)
		assert np.allclose(results_1, expected, rtol=1e-3)
		results_2 = _distance_matrix_d_magpie(embs, embs, multiprocessing, n_processes)
		assert np.allclose(results_2, expected, rtol=1e-3)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes, kwargs",
		[
			(False, None, {}),
			(True, N_PROCESSES, {"args_emb": {"k": 100}}),
			(False, None, {"args_emb": {"k": 100, "return_row_data": True}}),
			(
				True,
				None,
				{
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": True,
					}
				},
			),
			(
				False,
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
		],
	)
	def test_distance_matrix_d_pdd(
		self,
		prepare_four_xtals: tuple[Structure, Structure, Crystal, Crystal, dict],
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test _distance_matrix_d_pdd."""
		xtals = prepare_four_xtals
		embs = [
			xtal.get_embedding("pdd", **(kwargs.get("args_emb", {}))) for xtal in xtals
		]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_pdd(
					embs[i], embs[j], **(kwargs.get("args_dist", {}))
				)
		results_1 = _distance_matrix_d_pdd(
			embs, None, multiprocessing, n_processes, **(kwargs.get("args_dist", {}))
		)
		assert np.allclose(results_1, expected, rtol=1e-3)
		results_2 = _distance_matrix_d_pdd(
			embs, embs, multiprocessing, n_processes, **(kwargs.get("args_dist", {}))
		)
		assert np.allclose(results_2, expected, rtol=1e-3)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"args_emb": {"k": 100}},
			{"args_dist": {"metric": "chebyshev", "low_memory": False}},
			{
				"args_emb": {"k": 100},
				"args_dist": {"metric": "chebyshev", "low_memory": False},
			},
		],
	)
	def test_distance_matrix_d_amd(
		self,
		prepare_four_xtals: tuple[Structure, Structure, Crystal, Crystal, dict],
		kwargs: dict,
	):
		"""Test _distance_matrix_d_amd."""
		xtals = prepare_four_xtals
		embs = [
			xtal.get_embedding("amd", **(kwargs.get("args_emb", {}))) for xtal in xtals
		]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_amd(
					embs[i], embs[j], **(kwargs.get("args_dist", {}))
				)
		results_1 = _distance_matrix_d_amd(embs, None, **(kwargs.get("args_dist", {})))
		assert np.allclose(results_1, expected, rtol=1e-3)
		results_2 = _distance_matrix_d_amd(embs, embs, **(kwargs.get("args_dist", {})))
		assert np.allclose(results_2, expected, rtol=1e-3)

	@pytest.mark.parametrize(
		"multiprocessing, n_processes, kwargs",
		[
			(False, None, {}),
			(True, N_PROCESSES, {"metric": "mod_petti"}),
			(True, N_PROCESSES, {"metric": "fast", "return_assignments": True}),
			(True, None, {}),
		],
	)
	def test_distance_matrix_d_elmd(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test _distance_matrix_d_elmd."""
		xtals = prepare_four_xtals
		embs = [xtal.get_embedding("elmd") for xtal in xtals]
		expected = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				expected[i, j] = _d_elmd(embs[i], embs[j], **kwargs)
		results_1 = _distance_matrix_d_elmd(
			embs, None, multiprocessing, n_processes, **kwargs
		)
		assert np.allclose(results_1, expected, rtol=1e-3)
		results_2 = _distance_matrix_d_elmd(
			embs, embs, multiprocessing, n_processes, **kwargs
		)
		assert np.allclose(results_2, expected, rtol=1e-3)

	@pytest.mark.parametrize(
		"distance_name, multiprocessing, n_processes, kwargs",
		[
			("smat", False, None, {}),
			("smat", True, N_PROCESSES, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}),
			("comp", False, None, {}),
			("comp", True, None, {}),
			("wyckoff", False, None, {}),
			("wyckoff", True, N_PROCESSES, {}),
			("magpie", False, None, {}),
			("magpie", True, None, {}),
			("pdd", False, None, {}),
			("pdd", True, N_PROCESSES, {"k": 100}),
			("pdd", False, None, {"k": 100, "return_row_data": True}),
			("amd", True, None, {}),
			("amd", False, None, {"k": 100}),
			("elmd", False, None, {}),
			("elmd", True, N_PROCESSES, {}),
		],
	)
	def test_compute_embeddings(
		self,
		prepare_four_xtals: tuple[Crystal, Crystal, Crystal, Crystal],
		distance_name: str,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test compute_embeddings function."""
		xtals = prepare_four_xtals
		embs_1 = _compute_embeddings(
			distance_name, xtals[0], multiprocessing, n_processes, **kwargs
		)
		embs_all = _compute_embeddings(
			distance_name, xtals, multiprocessing, n_processes, **kwargs
		)
		assert type(embs_1) is type(embs_all[0])
		assert isinstance(embs_all, list)
		assert len(embs_all) == 4

	@pytest.mark.parametrize(
		"distance_name, kwargs",
		[
			("smat", {}),
			("smat", {"args_dist": {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}}),
			("comp", {}),
			("wyckoff", {}),
			("magpie", {}),
			("pdd", {}),
			("pdd", {"args_emb": {"k": 100}}),
			("pdd", {"args_emb": {"k": 100, "return_row_data": True}}),
			(
				"pdd",
				{
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": True,
					}
				},
			),
			(
				"pdd",
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
			("amd", {}),
			("amd", {"args_emb": {"k": 100}}),
			(
				"amd",
				{
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				"amd",
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			("elmd", {}),
			("elmd", {"args_dist": {"metric": "mod_petti"}}),
			("elmd", {"args_dist": {"metric": "fast", "return_assignments": True}}),
		],
	)
	def test_distance(
		self,
		prepare: tuple[Structure, Structure, Crystal, Crystal, dict],
		distance_name: str,
		kwargs: dict,
	):
		"""Test distance."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		emb_1 = _compute_embeddings(
			distance_name, xtal_1, False, **(kwargs.get("args_emb", {}))
		)
		emb_2 = _compute_embeddings(
			distance_name, xtal_2, False, **(kwargs.get("args_emb", {}))
		)
		d = [None for _ in range(6)]
		d[0] = distance(distance_name, struc_1, struc_2, **kwargs)
		d[1], _, _ = distance(distance_name, struc_1, struc_2, True, **kwargs)
		d[2] = distance(distance_name, xtal_1, xtal_2, **kwargs)
		d[3], _, _ = distance(distance_name, xtal_1, xtal_2, True, **kwargs)
		d[4] = distance(distance_name, emb_1, emb_2, **kwargs)
		d[5], _, _ = distance(distance_name, emb_1, emb_2, True, **kwargs)
		if distance_name in BINARY_DISTANCES:
			for di in d:
				assert di == expected[distance_name]
		else:
			for di in d:
				assert di == approx(expected[distance_name], rel=1e-3)

	@pytest.mark.parametrize(
		"distance_name, multiprocessing, n_processes, kwargs",
		[
			("smat", False, None, {}),
			("smat", True, N_PROCESSES, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}),
			("comp", False, None, {}),
			("comp", True, None, {}),
			("wyckoff", False, None, {}),
			("wyckoff", True, N_PROCESSES, {}),
			("magpie", False, None, {}),
			("magpie", True, None, {}),
			("pdd", False, None, {}),
			("pdd", True, N_PROCESSES, {"args_emb": {"k": 100}}),
			("pdd", False, None, {"args_emb": {"k": 100, "return_row_data": True}}),
			(
				"pdd",
				True,
				None,
				{
					"args_dist": {
						"metric": "chebyshev",
						"backend": "multiprocessing",
						"n_jobs": 2,
						"verbose": True,
					}
				},
			),
			(
				"pdd",
				False,
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
			("amd", True, N_PROCESSES, {}),
			("amd", False, None, {"args_emb": {"k": 100}}),
			(
				"amd",
				True,
				None,
				{
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			(
				"amd",
				False,
				None,
				{
					"args_emb": {"k": 100},
					"args_dist": {"metric": "chebyshev", "low_memory": False},
				},
			),
			("elmd", False, None, {}),
			("elmd", True, N_PROCESSES, {"args_dist": {"metric": "mod_petti"}}),
			(
				"elmd",
				False,
				None,
				{"args_dist": {"metric": "fast", "return_assignments": True}},
			),
		],
	)
	def test_distance_matrix(
		self,
		prepare_d_mtx: tuple[Structure, Structure, Structure, Structure, dict],
		distance_name: str,
		multiprocessing: bool,
		n_processes: int | None,
		kwargs: dict,
	):
		"""Test distance_matrix."""
		struc_1, struc_2, struc_3, struc_4, expected = prepare_d_mtx
		structures = [struc_1, struc_2, struc_3, struc_4]
		xtals = [Crystal.from_Structure(s) for s in structures]
		embs = [
			_compute_embeddings(
				distance_name,
				xtal,
				multiprocessing,
				n_processes,
				**(kwargs.get("args_emb", {})),
			)
			for xtal in xtals
		]
		matrices = [None for _ in range(12)]
		matrices[0] = distance_matrix(
			distance_name, structures, None, multiprocessing, n_processes, **kwargs
		)
		matrices[1], _, times_1 = distance_matrix(
			distance_name,
			structures,
			None,
			multiprocessing,
			n_processes,
			True,
			**kwargs,
		)
		matrices[2] = distance_matrix(
			distance_name,
			structures,
			structures,
			multiprocessing,
			n_processes,
			**kwargs,
		)
		matrices[3], _, _, times_2 = distance_matrix(
			distance_name,
			structures,
			structures,
			multiprocessing,
			n_processes,
			True,
			**kwargs,
		)
		matrices[4] = distance_matrix(
			distance_name, xtals, None, multiprocessing, n_processes, **kwargs
		)
		matrices[5], _, _ = distance_matrix(
			distance_name, xtals, None, multiprocessing, n_processes, True, **kwargs
		)
		matrices[6] = distance_matrix(
			distance_name, xtals, xtals, multiprocessing, n_processes, **kwargs
		)
		matrices[7], _, _, _ = distance_matrix(
			distance_name, xtals, xtals, multiprocessing, n_processes, True, **kwargs
		)
		matrices[8] = distance_matrix(
			distance_name, embs, None, multiprocessing, n_processes, **kwargs
		)
		matrices[9], _, _ = distance_matrix(
			distance_name, embs, None, multiprocessing, n_processes, True, **kwargs
		)
		matrices[10] = distance_matrix(
			distance_name, embs, embs, multiprocessing, n_processes, **kwargs
		)
		matrices[11], _, _, _ = distance_matrix(
			distance_name, embs, embs, multiprocessing, n_processes, True, **kwargs
		)
		if distance_name in BINARY_DISTANCES:
			for d_mtx in matrices:
				assert np.all(d_mtx == expected[distance_name])
		else:
			for d_mtx in matrices:
				assert d_mtx == approx(expected[distance_name], rel=1e-3)
		assert "emb_1" in times_1
		assert "d_mtx" in times_1
		assert "emb_1" in times_2
		assert "emb_2" in times_2
		assert "d_mtx" in times_2
