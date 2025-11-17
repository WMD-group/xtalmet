"""This module defines the Crystal class to store information about a single crystal."""

import warnings
from collections.abc import Sequence

import amd
import numpy as np
from amd import periodicset_from_pymatgen_structure
from ase import Atoms
from matminer.featurizers.composition.composite import ElementProperty
from numpy.typing import ArrayLike
from pymatgen.core import Composition, Lattice, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import CompositionLike

from .constants import (
	DIST_WO_EMB,
	TYPE_EMB_ALL,
	TYPE_EMB_AMD,
	TYPE_EMB_COMP,
	TYPE_EMB_ELMD,
	TYPE_EMB_MAGPIE,
	TYPE_EMB_PDD,
	TYPE_EMB_WYCKOFF,
)


class Crystal(Structure):
	"""Container for a single crystal structure."""

	def __init__(
		self,
		lattice: ArrayLike | Lattice,
		species: Sequence[CompositionLike],
		coords: Sequence[ArrayLike],
		charge: float | None = None,
		validate_proximity: bool = False,
		to_unit_cell: bool = False,
		coords_are_cartesian: bool = False,
		site_properties: dict | None = None,
		labels: Sequence[str | None] | None = None,
		properties: dict | None = None,
	) -> None:
		"""Initialize a Crystal object.

		The parameters are the same as those used in the __init__() method of the
		pymatgen.core.Structure class.

		Args:
			lattice (Lattice/3x3 array): The lattice, either as a
				pymatgen.core.Lattice or
				simply as any 2D array. Each row should correspond to a lattice
				vector. e.g. [[10,0,0], [20,10,0], [0,0,30]] specifies a
				lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
			species ([Species]): Sequence of species on each site. Can take in
				flexible input, including:

				i.  A sequence of element / species specified either as string
					symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
					e.g. (3, 56, ...) or actual Element or Species objects.

				ii. List of dict of elements/species and occupancies, e.g.
					[{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
					disordered structures.
			coords (Nx3 array): list of fractional/Cartesian coordinates of
				each species.
			charge (int): overall charge of the structure. Defaults to behavior
				in SiteCollection where total charge is the sum of the oxidation
				states.
			validate_proximity (bool): Whether to check if there are sites
				that are less than 0.01 Ang apart. Defaults to False.
			to_unit_cell (bool): Whether to map all sites into the unit cell,
				i.e. fractional coords between 0 and 1. Defaults to False.
			coords_are_cartesian (bool): Set to True if you are providing
				coordinates in Cartesian coordinates. Defaults to False.
			site_properties (dict): Properties associated with the sites as a
				dict of sequences, e.g. {"magmom":[5, 5, 5, 5]}. The sequences
				have to be the same length as the atomic species and
				fractional_coords. Defaults to None for no properties.
			labels (list[str]): Labels associated with the sites as a
				list of strings, e.g. ['Li1', 'Li2']. Must have the same
				length as the species and fractional coords. Defaults to
				None for no labels.
			properties (dict): Properties associated with the whole structure.
				Will be serialized when writing the structure to JSON or YAML but is
				lost when converting to other formats.

		Note:
			The descriptions for args are copied from pymatgen.core.Structure class.
		"""
		super().__init__(
			lattice,
			[
				specie.element if isinstance(specie, Species) else specie
				for specie in species
			],  # not considering oxidation states
			coords,
			charge,
			validate_proximity,
			to_unit_cell,
			coords_are_cartesian,
			site_properties,
			labels,
			properties,
		)

	@classmethod
	def from_Structure(cls, structure: Structure) -> "Crystal":
		"""Create a Crystal object from a pymatgen Structure object.

		Args:
			structure (Structure): A pymatgen Structure object.

		Returns:
			Crystal: A Crystal object created from the Structure.
		"""
		# internally calls __init__
		return cls(
			lattice=structure.lattice,
			species=structure.species,
			coords=structure.frac_coords,
			charge=structure._charge,
			validate_proximity=False,
			to_unit_cell=False,
			coords_are_cartesian=False,
			site_properties=structure.site_properties,
			labels=structure.labels,
			properties=structure.properties,
		)

	def _get_emb_d_comp(self) -> TYPE_EMB_COMP:
		"""Get the composition of the crystal as a tuple of (element, count).

		Embedding for d_comp.

		Returns:
			TYPE_EMB_COMP: A tuple of elements and their counts (divided by gcd).
		"""
		composition_unnorm = [
			(elem, int(count))
			for elem, count in dict(sorted(self.composition.as_dict().items())).items()
		]
		gcd = np.gcd.reduce([count for _, count in composition_unnorm]).item()
		composition = tuple((elem, count // gcd) for elem, count in composition_unnorm)
		return composition

	def _get_emb_d_wyckoff(self) -> TYPE_EMB_WYCKOFF:
		"""Get the Wyckoff representation of the crystal.

		Embedding for d_wyckoff.

		Returns:
			TYPE_EMB_WYCKOFF: A tuple containing the space group number and a tuple of
			Wyckoff letters.

		Raises:
			Exception: an exception from SpacegroupAnalyzer.
		"""
		sga = SpacegroupAnalyzer(self)
		sym = sga.get_symmetrized_structure()
		sg = sga.get_space_group_number()
		wyckoff_letters = sorted(
			[x[-1] for x in sym.wyckoff_symbols]
		)  # Don't use sym.wyckoff_letters
		return sg, tuple(wyckoff_letters)

	def _get_emb_d_magpie(self) -> TYPE_EMB_MAGPIE:
		"""Get the magpie embedding of the crystal.

		Embedding for d_magpie. Not influenced by oxidation states.

		Returns:
			TYPE_EMB_MAGPIE: Magpie feature vector of the crystal.

		References:
			- Ward et al., (2016). A general-purpose machine learning framework for
			  predicting properties of inorganic materials. npj Computational Materials,
			  2(1), 1-7. https://doi.org/10.1038/npjcompumats.2016.28
		"""
		if not hasattr(self, "featurizer"):
			self.featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
		feature = self.featurizer.featurize(self.composition)
		return [float(x) for x in feature]

	def _get_emb_d_pdd(self, k: int = 100, **kwargs) -> TYPE_EMB_PDD:
		"""Get the pointwise distance distribution (PDD) of the crystal.

		Embedding for d_ppd.

		Args:
			k (int): Number of nearest neighbors to consider.
			**kwargs: Additional arguments for amd.PDD, except k.

		Returns:
			TYPE_EMB_PDD: PDD.

		Raises:
			Exception: an exception from periodicset_from_pymatgen_structure.

		References:
			- Widdowson et al., (2022). Resolving the data ambiguity for periodic
			  crystals. Advances in Neural Information Processing Systems, 35,
			  24625--24638. https://openreview.net/forum?id=4wrB7Mo9_OQ
		"""
		if "return_row_data" in kwargs and kwargs["return_row_data"] is True:
			warnings.warn(
				"return_row_data=True is not supported. Automatically set to False.",
				UserWarning,
				stacklevel=2,
			)
			kwargs["return_row_data"] = False

		return amd.PDD(periodicset_from_pymatgen_structure(self), k, **kwargs)

	def _get_emb_d_amd(self, k: int = 100) -> TYPE_EMB_AMD:
		"""Get the average minimum distance (AMD) of the crystal.

		Embedding for d_amd.

		Args:
			k (int): Number of nearest neighbors to consider. Also the embedding length.

		Returns:
			TYPE_EMB_AMD: AMD.

		Raises:
			Exception: an exception from periodicset_from_pymatgen_structure.

		References:
			- Widdson et al., (2022). Average Minimum Distances of periodic point sets -
			  foundational invariants for mapping periodic crystals. MATCH
			  Communications in Mathematical and in Computer Chemistry, 87(3), 529-559,
			  https://doi.org/10.46793/match.87-3.529W
		"""
		return amd.AMD(periodicset_from_pymatgen_structure(self), k)

	def _get_emb_d_elmd(self) -> TYPE_EMB_ELMD:
		"""Get the embedding for d_elmd.

		Embedding for d_elmd.

		Returns:
			TYPE_EMB_ELMD: Compositional formula as a string.
		"""
		return self.composition.reduced_formula

	def get_embedding(self, distance: str, **kwargs) -> TYPE_EMB_ALL:
		"""Get the embedding of the crystal based on the specified distance metric.

		Args:
			distance (str): The distance metric to use.
			**kwargs: Additional arguments for embedding methods if needed.

		Returns:
			TYPE_EMB_ALL: The embedding corresponding to the specified distance
			metric.

		Raises:
			ValueError: If an unsupported distance metric is provided.

		Note:
			For "smat" distance, the embedding is the Crystal object itself.
		"""
		if distance == "comp":
			return self._get_emb_d_comp()
		elif distance == "wyckoff":
			return self._get_emb_d_wyckoff()
		elif distance == "magpie":
			return self._get_emb_d_magpie()
		elif distance == "pdd":
			return self._get_emb_d_pdd(**kwargs)
		elif distance == "amd":
			return self._get_emb_d_amd(**kwargs)
		elif distance == "elmd":
			return self._get_emb_d_elmd()
		elif distance in DIST_WO_EMB:
			return self
		else:
			raise ValueError(f"Unsupported distance metric: {distance}")

	def get_composition_pymatgen(self) -> Composition:
		"""Get the pymatgen composition of the crystal.

		Called when screening using SMACT or E_hull.

		Returns:
			Composition: Pymatgen Composition object.
		"""
		return self.composition

	def get_ase_atoms(self) -> Atoms:
		"""Get the ASE Atoms object of the crystal.

		Called when screening using E_hull. Not influenced by oxidation states.

		Returns:
			Atoms: ASE Atoms object.
		"""
		return AseAtomsAdaptor.get_atoms(self)
