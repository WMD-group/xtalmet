"""Test Materials Project entries across the pymatgen-core serialization split."""

import copy
import json
from types import SimpleNamespace

import pytest
from monty.json import MontyDecoder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.entries.computed_entries import (
	ComputedEntry,
	ComputedStructureEntry,
	ConstantEnergyAdjustment,
)
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

from xtalmet.stability import StabilityCalculator


def mp_entry(formula, energy, identifier, module, structured=True, mixing=False):
	"""Build a small MSON entry with the metadata used by the live API."""
	composition = Composition(formula)
	compatibility = {
		"@module": (
			"pymatgen.analysis.compatibility"
			if module == "pymatgen.core.entries"
			else "pymatgen.entries.compatibility"
		),
		"@class": "MaterialsProject2020Compatibility",
	}
	if mixing:
		matcher = StructureMatcher(stol=0.42).as_dict()
		if module == "pymatgen.core.entries":
			matcher["@module"] = "pymatgen.core.structure_matcher"
			matcher["comparator"]["@module"] = "pymatgen.core.structure_matcher"
		compatibility = {
			"@module": (
				"pymatgen.analysis.compatibility.mixing_scheme"
				if module == "pymatgen.core.entries"
				else "pymatgen.entries.mixing_scheme"
			),
			"@class": "MaterialsProjectDFTMixingScheme",
			"structure_matcher": matcher,
			"compat_1": compatibility,
		}
	parameters = {
		"run_type": "GGA",
		"hubbards": {},
		"potcar_spec": [
			{
				"titel": f"PAW_PBE {element} 01Jan2000",
				"hash": "unused-for-symbol-check",
				"@module": "emmet.core.vasp.calculation",
				"@class": "PotcarSpec",
			}
			for element in composition
		],
	}
	kwargs = {
		"energy": energy,
		"parameters": parameters,
		"data": {"run_type": "GGA", "oxide_type": "oxide"},
		"entry_id": identifier,
		"energy_adjustments": [ConstantEnergyAdjustment(0.5, cls=compatibility)],
	}
	if structured:
		species = [
			str(element)
			for element, amount in composition.items()
			for _ in range(int(amount))
		]
		structure = Structure(
			Lattice.cubic(5),
			species,
			[[i / len(species)] * 3 for i in range(len(species))],
		)
		entry = ComputedStructureEntry(structure, **kwargs)
	else:
		entry = ComputedEntry(composition, **kwargs)
	result = entry.as_dict()
	result["@module"] = module
	result["energy_adjustments"][0]["@module"] = module
	return result


@pytest.mark.parametrize(
	"module", ["pymatgen.entries.computed_entries", "pymatgen.core.entries"]
)
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("mixing", [False, True])
def test_mp_entry_decoding(module, structured, mixing):
	"""Keep structures, corrections and POTCAR fields across both module paths."""
	from xtalmet.stability import _decode_mp_entry

	raw = mp_entry("SiO2", -20, "oxide", module, structured, mixing)
	before = copy.deepcopy(raw)
	decoded = _decode_mp_entry(raw)

	assert isinstance(decoded, ComputedStructureEntry if structured else ComputedEntry)
	assert decoded.uncorrected_energy == -20
	assert decoded.correction == 0.5
	assert decoded.entry_id == "oxide"
	assert decoded.data == raw["data"]
	assert decoded.parameters["potcar_spec"][0]["titel"] == "PAW_PBE Si 01Jan2000"
	assert isinstance(decoded.parameters["potcar_spec"][0], dict)
	if mixing:
		provenance = decoded.energy_adjustments[0].cls
		assert isinstance(provenance, MaterialsProjectDFTMixingScheme)
		assert isinstance(provenance.structure_matcher, StructureMatcher)
		assert provenance.structure_matcher.stol == 0.42
	if structured:
		assert decoded.structure == Structure.from_dict(raw["structure"])
	assert raw == before


@pytest.mark.parametrize(
	"module", ["pymatgen.entries.computed_entries", "pymatgen.core.entries"]
)
def test_mp_phase_diagram_from_raw_entries(module, monkeypatch):
	"""Apply compatibility and deduplication, then use uncorrected GGA energies."""
	entries = [
		mp_entry("Si", -5, "si", module),
		mp_entry("O2", -8, "oxygen", module),
		mp_entry("SiO2", -20, "oxide", module),
	]
	r2scan = mp_entry("SiO2", -25, "r2scan", module, mixing=True)
	r2scan["parameters"]["run_type"] = "r2SCAN"
	r2scan["data"]["run_type"] = "R2SCAN"
	gga_u = mp_entry("SiO2", -19, "gga-u", module)
	gga_u["parameters"]["run_type"] = "GGA+U"
	gga_u["data"]["run_type"] = "GGA_U"
	wrong_potcar = mp_entry("SiO2", -30, "wrong-potcar", module)
	wrong_potcar["parameters"]["potcar_spec"][0]["titel"] = "PAW_PBE Si_wrong 01Jan2000"
	response = [
		{"entries": {"GGA": entry}}
		for entry in [*entries, entries[2], r2scan, gga_u, wrong_potcar]
	]
	before = copy.deepcopy(response)

	def request(url, mp_decode=True):
		assert url == "materials/thermo/?_fields=entries&formula="
		if mp_decode:
			return json.loads(json.dumps(response), cls=MontyDecoder)
		return response

	monkeypatch.setattr(
		"xtalmet.stability.MPRester", lambda key: SimpleNamespace(request=request)
	)
	model = object()
	monkeypatch.setattr("xtalmet.stability.mace_mp", lambda **kwargs: model)
	calculator = StabilityCalculator(diagram="mp")

	actual = {
		(entry.composition.reduced_formula, entry.energy)
		for entry in calculator.ppd_mp.all_entries
	}
	assert actual == {("Si", -5), ("O2", -8), ("SiO2", -20), ("SiO2", -19)}
	assert len(calculator.ppd_mp.all_entries) == 4
	assert calculator.calculator is model
	assert response == before
