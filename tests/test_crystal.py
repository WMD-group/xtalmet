"""Test the Crystal class."""

import os

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError

from xtalmet.crystal import Crystal


@pytest.fixture(scope="module", params=["wz-ZnO", "rs-ZnO", "wz-GaN", "Bi2Te3"])
def prepare_Structure(request) -> Structure:
	"""Prepare a pymatgen Structure object for testing."""
	structure = Structure.from_file(
		os.path.join(os.path.dirname(__file__), f"data/{request.param}.cif")
	)
	return structure


class TestCrystal:
	"""Test the Crystal class."""

	def test_init(self, prepare_Structure: Structure):
		"""Test __init__."""
		structure = prepare_Structure
		crystal = Crystal(
			lattice=structure.lattice,
			species=structure.species,
			coords=structure.frac_coords,
		)
		# Ensure that the Crystal instance correctly inherits the Structure instance.
		for key in structure.__dict__:
			if key != "_sites":
				assert crystal.__dict__[key] == structure.__dict__[key]
			else:
				for site1, site2 in zip(
					crystal.__dict__[key], structure.__dict__[key], strict=True
				):
					assert site1.species.is_element
					assert site1.lattice == site2.lattice
					assert all(site1.frac_coords == site2.frac_coords)
					assert site1.properties == site2.properties

	def test_from_Structure(self, prepare_Structure: Structure):
		"""Test from_Structure."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		# Ensure that the Crystal instance correctly inherits the Structure instance.
		for key in structure.__dict__:
			if key != "_sites":
				assert crystal.__dict__[key] == structure.__dict__[key]
			else:
				for site1, site2 in zip(
					crystal.__dict__[key], structure.__dict__[key], strict=True
				):
					assert site1.species.is_element
					assert site1.lattice == site2.lattice
					assert all(site1.frac_coords == site2.frac_coords)
					assert site1.properties == site2.properties

	def test_get_emb_d_comp(self, prepare_Structure: Structure):
		"""Test _get_emb_d_comp."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		composition_tuple = crystal._get_emb_d_comp()
		assert isinstance(composition_tuple, tuple)
		assert all(
			isinstance(item, tuple) and len(item) == 2 for item in composition_tuple
		)

	def test_get_emb_d_wyckoff(self, prepare_Structure: Structure):
		"""Test _get_emb_d_wyckoff."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		try:
			wyckoff = crystal._get_emb_d_wyckoff()
			assert isinstance(wyckoff, tuple)
			assert isinstance(wyckoff[0], int)
			assert isinstance(wyckoff[1], tuple)
			assert all(isinstance(item, str) for item in wyckoff[1])
		except Exception as e:
			# If an exception is raised, ensure it's a known issue.
			assert isinstance(e, SymmetryUndeterminedError)

	def test_get_emb_d_magpie(self, prepare_Structure: Structure):
		"""Test _get_emb_d_magpie."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		magpie = crystal._get_emb_d_magpie()
		assert isinstance(magpie, list)
		assert all(isinstance(item, float) for item in magpie)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"k": 100},
			{"k": 200, "return_row_data": True},
		],
	)
	def test__get_emb_d_pdd(self, prepare_Structure: Structure, kwargs: dict):
		"""Test _get_emb_d_pdd."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		try:
			pdd = crystal._get_emb_d_pdd(**kwargs)
			assert isinstance(pdd, np.ndarray)
			assert pdd.ndim == 2
			if "k" in kwargs:
				assert pdd.shape[1] == kwargs["k"] + 1
			else:
				assert pdd.shape[1] == 100 + 1
		except Exception as e:
			# If an exception is raised, ensure it's a known issue.
			assert isinstance(e, SymmetryUndeterminedError)

	@pytest.mark.parametrize(
		"kwargs",
		[
			{},
			{"k": 200},
		],
	)
	def test_get_emb_d_amd(self, prepare_Structure: Structure, kwargs: dict):
		"""Test _get_emb_d_amd."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		try:
			amd = crystal._get_emb_d_amd(**kwargs)
			assert isinstance(amd, np.ndarray)
			assert amd.ndim == 1
			if "k" in kwargs:
				assert amd.size == kwargs["k"]
			else:
				assert amd.size == 100
		except Exception as e:
			# If an exception is raised, ensure it's a known issue.
			assert isinstance(e, SymmetryUndeterminedError)

	def test_get_emb_d_elmd(self, prepare_Structure: Structure):
		"""Test _get_emb_d_elmd."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		elmd = crystal._get_emb_d_elmd()
		assert isinstance(elmd, str)
		assert elmd == structure.composition.reduced_formula

	@pytest.mark.parametrize(
		"distance, kwargs",
		[
			("smat", {}),
			("comp", {}),
			("wyckoff", {}),
			("magpie", {}),
			("pdd", {}),
			("pdd", {"k": 150}),
			("amd", {}),
			("amd", {"k": 150}),
			("elmd", {}),
		],
	)
	def test_get_embedding(
		self, prepare_Structure: Structure, distance: str, kwargs: dict
	):
		"""Test get_embedding."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		embedding = crystal.get_embedding(distance, **kwargs)
		assert embedding is not None

	def test_get_composition_pymatgen(self, prepare_Structure: Structure):
		"""Test get_composition_pymatgen."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		composition = crystal.get_composition_pymatgen()
		assert composition == structure.composition

	def test_get_ase_atoms(self, prepare_Structure: Structure):
		"""Test get_ase_atoms."""
		structure = prepare_Structure
		crystal = Crystal.from_Structure(structure)
		atoms = crystal.get_ase_atoms()
		assert isinstance(atoms, Atoms)
