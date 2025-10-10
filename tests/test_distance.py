"""Test distance functions."""

import os

import pytest
from pymatgen.core import Structure
from pytest import approx

from xtalmet.crystal import Crystal
from xtalmet.distance import d_amd, d_comp, d_magpie, d_pdd, d_smat, d_wyckoff


@pytest.fixture(
	scope="module",
	params=[
		(
			"wz-ZnO",
			"rs-ZnO",
			{
				"d_smat": 1.0,
				"d_comp": 0.0,
				"d_wyckoff": 1.0,
				"d_magpie": 0.000e00,
				"d_amd": 1.042e00,
				"d_pdd": 1.042e00,
			},
		),
		(
			"wz-ZnO",
			"wz-GaN",
			{
				"d_smat": 1.0,
				"d_comp": 1.0,
				"d_wyckoff": 0.0,
				"d_magpie": 6.298e02,
				"d_amd": 9.684e-02,
				"d_pdd": 9.684e-02,
			},
		),
		(
			"wz-ZnO",
			"Bi2Te3",
			{
				"d_smat": 1.0,
				"d_comp": 1.0,
				"d_wyckoff": 1.0,
				"d_magpie": 1.070e03,
				"d_amd": 3.240e00,
				"d_pdd": 3.276e00,
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


class TestDistance:
	"""Test distance functions."""

	@pytest.mark.parametrize("kwargs", [{}, {"ltol": 0.3, "stol": 0.4, "angle_tol": 6}])
	def test_d_smat(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test d_smat."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_smat(xtal_1, xtal_2, **kwargs) == expected["d_smat"]
		assert d_smat(struc_1, struc_2, **kwargs) == expected["d_smat"]

	def test_d_comp(self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]):
		"""Test d_comp."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_comp(struc_1, struc_2) == expected["d_comp"]
		assert d_comp(xtal_1, xtal_2) == expected["d_comp"]
		assert (
			d_comp(xtal_1.get_composition_tuple(), xtal_2.get_composition_tuple())
			== expected["d_comp"]
		)

	def test_d_wyckoff(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]
	):
		"""Test d_wyckoff."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_wyckoff(struc_1, struc_2) == expected["d_wyckoff"]
		assert d_wyckoff(xtal_1, xtal_2) == expected["d_wyckoff"]
		assert (
			d_wyckoff(xtal_1.get_wyckoff(), xtal_2.get_wyckoff())
			== expected["d_wyckoff"]
		)

	def test_d_magpie(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict]
	):
		"""Test d_magpie."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_magpie(struc_1, struc_2) == approx(expected["d_magpie"], rel=1e-3)
		assert d_magpie(xtal_1, xtal_2) == approx(expected["d_magpie"], rel=1e-3)
		assert d_magpie(xtal_1.get_magpie(), xtal_2.get_magpie()) == approx(
			expected["d_magpie"], rel=1e-3
		)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"emb": {"k": 100}},
			{"emb": {"k": 100, "return_row_data": True}},
			{
				"dist": {
					"metric": "chebyshev",
					"backend": "multiprocessing",
					"n_jobs": 2,
					"verbose": True,
				}
			},
			{
				"emb": {"k": 100},
				"dist": {
					"metric": "chebyshev",
					"backend": "multiprocessing",
					"n_jobs": None,
					"verbose": False,
				},
			},
		],
	)
	def test_d_pdd(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test d_pdd."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_pdd(struc_1, struc_2, **kwargs) == approx(expected["d_pdd"], rel=1e-3)
		assert d_pdd(xtal_1, xtal_2, **kwargs) == approx(expected["d_pdd"], rel=1e-3)
		assert d_pdd(xtal_1.get_PDD(), xtal_2.get_PDD(), **kwargs) == approx(
			expected["d_pdd"], rel=1e-3
		)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"emb": {"k": 100}},
			{"dist": {"metric": "chebyshev", "low_memory": False}},
			{"emb": {"k": 100}, "dist": {"metric": "chebyshev", "low_memory": False}},
		],
	)
	def test_d_amd(
		self, prepare: tuple[Structure, Structure, Crystal, Crystal, dict], kwargs: dict
	):
		"""Test d_amd."""
		struc_1, struc_2, xtal_1, xtal_2, expected = prepare
		assert d_amd(struc_1, struc_2, **kwargs) == approx(expected["d_amd"], rel=1e-3)
		assert d_amd(xtal_1, xtal_2, **kwargs) == approx(expected["d_amd"], rel=1e-3)
		assert d_amd(xtal_1.get_AMD(), xtal_2.get_AMD(), **kwargs) == approx(
			expected["d_amd"], rel=1e-3
		)
