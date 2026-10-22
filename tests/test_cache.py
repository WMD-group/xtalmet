"""Regression coverage for cache identities, reference settings, and shared work."""

import importlib
import importlib.util
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.analysis.structure_matcher import AbstractComparator
from pymatgen.core import Element

from xtalmet._cache import (
	MP20_REFERENCE_REVISION,
	_cache_path,
	_embedding_settings,
	_fingerprint,
	_read_record,
	_write_record,
)
from xtalmet.crystal import Crystal
from xtalmet.distance import distance_matrix
from xtalmet.evaluator import Evaluator
from xtalmet.stability import StabilityCalculator


@pytest.fixture
def structures():
	"""Distinct geometries with repeated and different compositions."""
	return [
		Crystal(np.eye(3) * size, [element], [[0, 0, 0]])
		for size, element in ((3, "Si"), (4, "O"), (5, "Si"))
	]


def count_work(monkeypatch):
	"""Count real embedding and matrix computations at the evaluator boundary."""
	module = importlib.import_module("xtalmet.evaluator")
	calls = {"embeddings": 0, "matrices": 0}
	for name, key in (
		("_compute_embeddings", "embeddings"),
		("distance_matrix", "matrices"),
	):
		original = getattr(module, name)

		def wrapped(*args, _original=original, _key=key, **kwargs):
			calls[_key] += 1
			return _original(*args, **kwargs)

		monkeypatch.setattr(module, name, wrapped)
	return calls


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_shared_embeddings(structures, tmp_path, monkeypatch, disk, parallel):
	"""U/N share one preparation; uncached subsequent evaluations recompute."""
	evaluator = Evaluator(
		uniqueness=True, novelty=True, distance="comp", ref_xtals=structures[:1]
	)
	calls = count_work(monkeypatch)
	directory = str(tmp_path) if disk else None
	first = evaluator.evaluate(structures, directory, parallel, 2)
	assert calls == {"embeddings": 1, "matrices": 2}
	assert first[2]["nov_emb"] == 0
	second = evaluator.evaluate(structures, directory, parallel, 2)
	assert calls == (
		{"embeddings": 1, "matrices": 2} if disk else {"embeddings": 2, "matrices": 4}
	)
	np.testing.assert_array_equal(first[1], second[1])
	if disk:
		assert (
			second[2]["uni_emb"]
			== second[2]["uni_d_mtx"]
			== second[2]["nov_d_mtx"]
			== 0
		)
		assert len(list((tmp_path / "cache" / "format-1").glob("*/*.pkl.gz"))) == 3


@pytest.mark.parametrize("change", ["order", "geometry", "composition", "count"])
def test_input_identity(structures, tmp_path, monkeypatch, change):
	"""Same-sized changed inputs and reordering must not reuse old artifacts."""
	evaluator = Evaluator(uniqueness=True, distance="comp")
	calls = count_work(monkeypatch)
	baseline = evaluator.evaluate(structures, str(tmp_path))[1]
	changed = [item.copy() for item in structures]
	if change == "order":
		changed.reverse()
	elif change == "geometry":
		changed[0].translate_sites([0], [0.1, 0, 0])
	elif change == "composition":
		changed[0].replace(0, "Li")
	else:
		changed.append(changed[0].copy())
	evaluator.evaluate(changed, str(tmp_path))
	assert calls == {"embeddings": 2, "matrices": 2}
	np.testing.assert_array_equal(
		evaluator.evaluate(structures, str(tmp_path))[1], baseline
	)
	assert calls == {"embeddings": 2, "matrices": 2}


def test_reference_identity_and_order(structures, tmp_path, monkeypatch):
	"""Reference changes invalidate novelty while retaining generated work and U."""
	evaluator = Evaluator(
		uniqueness=True, novelty=True, distance="comp", ref_xtals=structures[:2]
	)
	calls = count_work(monkeypatch)
	evaluator.evaluate(structures, str(tmp_path))
	evaluator.ref_xtals.reverse()
	evaluator.evaluate(structures, str(tmp_path))
	assert calls == {"embeddings": 1, "matrices": 3}
	evaluator.ref_xtals = [structures[0].get_embedding("comp")]
	evaluator.evaluate(structures, str(tmp_path))
	assert calls == {"embeddings": 1, "matrices": 4}


def test_embedding_and_distance_settings(structures, tmp_path, monkeypatch):
	"""Distance settings reuse embeddings; changed neighbor count recomputes both."""
	evaluator = Evaluator(uniqueness=True, distance="amd", args_emb={"k": 4})
	calls = count_work(monkeypatch)
	evaluator.evaluate(structures, str(tmp_path))
	evaluator.args_dist = {"metric": "euclidean"}
	evaluator.evaluate(structures, str(tmp_path))
	assert calls == {"embeddings": 1, "matrices": 2}
	evaluator.args_emb = {"k": 5}
	evaluator.evaluate(structures, str(tmp_path))
	assert calls == {"embeddings": 2, "matrices": 3}


def test_validity_settings_and_aggregation(structures, tmp_path, monkeypatch):
	"""A validator's actual settings affect its cache; aggregation does not."""
	evaluator = Evaluator(validity=["structure"])
	evaluator.evaluate(structures, str(tmp_path))
	validator = evaluator.validator.validators["structure"]
	validator.threshold_volume = 1000
	np.testing.assert_array_equal(
		evaluator.evaluate(structures, str(tmp_path))[1], [0, 0, 0]
	)
	validator.threshold_volume = 0.1
	monkeypatch.setattr(
		type(validator), "validate", lambda *a: pytest.fail("cache miss")
	)
	evaluator._setup_aggregation("ave", {"validity": 2})
	np.testing.assert_array_equal(
		evaluator.evaluate(structures, str(tmp_path))[1], [1, 1, 1]
	)


@pytest.mark.parametrize(
	"damage", ["gzip", "deflate", "version", "metadata", "shape", "payload"]
)
def test_invalid_cache_recomputed(structures, tmp_path, monkeypatch, damage):
	"""Broken records never supply scores, including valid pickles of wrong shape."""
	evaluator = Evaluator(uniqueness=True, distance="comp")
	expected = evaluator.evaluate(structures, str(tmp_path))[1]
	path = next(tmp_path.rglob("mtx_uni_comp.pkl.gz"))
	record = _read_record(path)
	if damage == "gzip":
		path.write_bytes(b"incomplete gzip")
	elif damage == "deflate":
		compressed = bytearray(path.read_bytes())
		compressed[21] ^= 0xFF
		path.write_bytes(compressed)
	elif damage == "metadata":
		_write_record(path, record["data"], {"wrong": True})
	elif damage == "shape":
		_write_record(path, np.zeros((1, 1)), record["metadata"])
	else:
		record["cache_format_version" if damage == "version" else "data"] = 999
		Evaluator._write_pickle_gz(None, record, str(path))
	calls = count_work(monkeypatch)
	with pytest.warns(UserWarning, match="Ignoring invalid cache"):
		actual = evaluator.evaluate(structures, str(tmp_path))[1]
	np.testing.assert_array_equal(actual, expected)
	assert calls == {"embeddings": 0, "matrices": 1}


def test_legacy_files_preserved(structures, tmp_path):
	"""Unverified old files remain byte-identical and do not enter evaluation."""
	path = tmp_path / "emb_comp.pkl.gz"
	Evaluator._write_pickle_gz(None, [(("Li", 1),)] * 3, str(path))
	before = path.read_bytes()
	actual = Evaluator(uniqueness=True, distance="comp").evaluate(
		structures, str(tmp_path)
	)[1]
	np.testing.assert_array_equal(actual, [1, 1, 0])
	assert path.read_bytes() == before


@pytest.mark.parametrize(
	"metric,args",
	[
		("amd", {"k": 4}),
		("pdd", {"k": 4, "collapse": False}),
		("wyckoff", {"symprec": 0.1}),
		("elmd+amd", {"amd": {"k": 4}}),
	],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_mp20_custom_settings(
	structures, tmp_path, monkeypatch, metric, args, parallel
):
	"""Custom MP20 reference preparation uses structures and the requested settings."""
	path = tmp_path / "train_smat.pkl.gz"
	Evaluator._write_pickle_gz(None, structures, str(path))
	requests = []

	def download(**kwargs):
		requests.append(kwargs)
		return str(path)

	monkeypatch.setattr("xtalmet.evaluator.hf_hub_download", download)
	with pytest.warns(UserWarning, match="Non-default args_emb"):
		evaluator = Evaluator(
			novelty=True,
			distance=metric,
			ref_xtals="mp20",
			args_emb=args,
			multiprocessing=parallel,
			n_processes=2,
		)
	assert len(requests) == 1
	assert requests[0]["filename"] == "mp20/train/train_smat.pkl.gz"
	assert requests[0]["revision"] == MP20_REFERENCE_REVISION
	expected = distance_matrix(metric, structures, structures, args_emb=args)
	np.testing.assert_allclose(
		distance_matrix(metric, evaluator.ref_xtals, evaluator.ref_xtals), expected
	)
	monkeypatch.setattr(
		"xtalmet.evaluator._compute_embeddings",
		lambda *a, **k: pytest.fail("reference recomputed"),
	)
	assert len(evaluator.ref_xtals) == 3


@pytest.mark.parametrize(
	"metric,args",
	[
		("amd", {"k": 100}),
		("wyckoff", {"symprec": 0.01, "angle_tolerance": 5.0}),
		("pdd", {"k": 100, "collapse_tol": 1e-4}),
		("elmd+amd", {"amd": {"k": 100}}),
	],
)
def test_mp20_explicit_defaults(tmp_path, monkeypatch, metric, args):
	"""Explicit default values and non-default distance settings use published data."""
	path = tmp_path / "published.pkl.gz"
	# A sentinel-only list reaches the final validity check without backend work.
	from xtalmet.distance import _EmbeddingError

	Evaluator._write_pickle_gz(
		None, [_EmbeddingError(ValueError("bad reference"))], str(path)
	)
	requested = []

	def download(**kwargs):
		requested.append(kwargs["filename"])
		return str(path)

	monkeypatch.setattr("xtalmet.evaluator.hf_hub_download", download)
	monkeypatch.setattr(
		"xtalmet.evaluator._compute_embeddings",
		lambda *a, **k: pytest.fail("recomputed"),
	)
	with (
		warnings.catch_warnings(record=True) as caught,
		pytest.raises(ValueError, match="valid reference"),
	):
		warnings.simplefilter("always")
		Evaluator(
			novelty=True,
			distance=metric,
			ref_xtals="mp20",
			args_emb=args,
			args_dist={"metric": "euclidean"},
		)
	assert not caught
	assert requested == [f"mp20/train/train_{metric}.pkl.gz"]


def test_reference_round_trip_and_mismatch(structures, tmp_path):
	"""Saved custom references retain configuration and reject different settings."""
	path = tmp_path / "refs.pkl.gz"
	original = Evaluator(
		novelty=True, distance="amd", ref_xtals=structures, args_emb={"k": 4}
	)
	original.save_reference_embeddings(path)
	loaded = Evaluator(
		novelty=True, distance="amd", ref_xtals=str(path), args_emb={"k": 4}
	)
	np.testing.assert_allclose(loaded.ref_xtals, original.ref_xtals)
	with pytest.raises(ValueError, match="metadata"):
		Evaluator(novelty=True, distance="amd", ref_xtals=str(path), args_emb={"k": 5})
	with pytest.raises(ValueError, match="metadata"):
		Evaluator(novelty=True, distance="comp", ref_xtals=str(path))
	with pytest.raises(ValueError, match="disabled"):
		Evaluator(uniqueness=True, distance="comp").save_reference_embeddings(path)
	Evaluator._write_pickle_gz(None, original.ref_xtals, str(path))
	with pytest.raises(ValueError, match="versioned embedding metadata"):
		Evaluator(novelty=True, distance="amd", ref_xtals=str(path), args_emb={"k": 4})


def test_stability_actual_inputs(structures, tmp_path, monkeypatch):
	"""Actual model and phase data control ehull reuse, independently of calibration."""
	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.calculator = SimpleNamespace(
		models=[torch.nn.Linear(1, 1)],
		head="default",
		model_type="MACE",
		device="cpu",
		energy_units_to_eV=1,
		length_units_to_A=1,
	)
	calculator.ppd_mp = PatchedPhaseDiagram(
		[PDEntry("Si", 0), PDEntry("O", 0), PDEntry("SiO2", -3)]
	)
	calculator.binary, calculator.threshold, calculator.intercept = True, 0.1, 0.4289
	calls = []
	calculator._ehull = lambda xtals: calls.append(True) or np.full(len(xtals), 0.05)
	monkeypatch.setattr("xtalmet.evaluator.StabilityCalculator", lambda **k: calculator)
	evaluator = Evaluator(stability="binary")
	evaluator.evaluate(structures, str(tmp_path))
	calculator.threshold = 0.01
	np.testing.assert_array_equal(
		evaluator.evaluate(structures, str(tmp_path))[1], [0, 0, 0]
	)
	assert len(calls) == 1
	with torch.no_grad():
		calculator.calculator.models[0].weight.add_(1)
	evaluator.evaluate(structures, str(tmp_path))
	assert len(calls) == 2
	patch = next(iter(calculator.ppd_mp.pds.values()))
	patch.qhull_data[0, -1] -= 0.1
	evaluator.evaluate(structures, str(tmp_path))
	assert len(calls) == 3
	calculator.ppd_mp = PatchedPhaseDiagram([PDEntry("Si", -1)])
	evaluator.evaluate(structures, str(tmp_path))
	assert len(calls) == 4


def test_mace_non_tensor_settings_invalidate_cache(tmp_path, monkeypatch):
	"""Real MACE settings outside state_dict change energy and invalidate ehull."""
	from e3nn import o3
	from mace import modules
	from mace.calculators import MACECalculator

	with torch.random.fork_rng():
		torch.manual_seed(1)
		model = modules.ScaleShiftMACE(
			r_max=3.0,
			num_bessel=2,
			num_polynomial_cutoff=3,
			max_ell=1,
			interaction_cls=modules.RealAgnosticResidualInteractionBlock,
			interaction_cls_first=modules.RealAgnosticInteractionBlock,
			num_interactions=2,
			num_elements=1,
			hidden_irreps=o3.Irreps("2x0e + 2x1o"),
			MLP_irreps=o3.Irreps("2x0e"),
			atomic_energies=np.array([0.0]),
			avg_num_neighbors=1.0,
			atomic_numbers=[14],
			correlation=1,
			gate=torch.nn.functional.silu,
			atomic_inter_scale=1.0,
			atomic_inter_shift=0.0,
		).double()
	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.calculator = MACECalculator(
		models=[model], device="cpu", default_dtype="float64"
	)
	calculator.ppd_mp = PatchedPhaseDiagram([PDEntry("Si", -1)])
	calculator.binary, calculator.intercept = False, 2
	monkeypatch.setattr("xtalmet.evaluator.StabilityCalculator", lambda **k: calculator)
	evaluator = Evaluator(stability="continuous")
	xtals = [Crystal(np.eye(3) * 10, ["Si", "Si"], [[0, 0, 0], [0.15, 0, 0]])]
	original = calculator._cache_identity()
	first = evaluator.evaluate(xtals, str(tmp_path))[1]
	assert calculator._cache_identity() == original
	model.interactions[0].avg_num_neighbors = 2.0
	calculator.calculator.reset()
	assert calculator._cache_identity() != original
	changed = evaluator.evaluate(xtals, str(tmp_path))[1]
	assert not np.array_equal(first, changed)
	np.testing.assert_array_equal(changed, evaluator.evaluate(xtals)[1])
	monkeypatch.setattr(calculator, "_ehull", lambda x: pytest.fail("cache miss"))
	np.testing.assert_array_equal(changed, evaluator.evaluate(xtals, str(tmp_path))[1])


def test_output_buffer_and_unidentifiable_settings(structures, tmp_path, monkeypatch):
	"""Cache hits never suppress output-buffer effects or fingerprint a callable."""
	out = np.full(3, -1.0)
	evaluator = Evaluator(
		uniqueness=True, distance="amd", args_emb={"k": 4}, args_dist={"out": out}
	)
	calls = count_work(monkeypatch)
	evaluator.evaluate(structures, str(tmp_path))
	assert (out >= 0).all()
	out.fill(-1)
	evaluator.evaluate(structures, str(tmp_path))
	assert (out >= 0).all()
	assert calls == {"embeddings": 1, "matrices": 2}
	assert not list(tmp_path.rglob("mtx_uni_amd.pkl.gz"))
	evaluator.args_dist = {"metric": lambda x, y: np.max(np.abs(x - y))}
	with pytest.warns(UserWarning, match="Disk cache disabled"):
		evaluator.evaluate(structures, str(tmp_path))
	assert calls == {"embeddings": 1, "matrices": 3}


def test_atomic_write_and_fingerprints(tmp_path, monkeypatch):
	"""A failed write preserves the old record; array hashing preserves order."""
	path = tmp_path / "cache.pkl.gz"
	_write_record(path, [1, 2], {})
	before = path.read_bytes()

	def fail(*args, **kwargs):
		raise RuntimeError("write interrupted")

	monkeypatch.setattr("xtalmet._cache.pickle.dump", fail)
	with pytest.raises(RuntimeError, match="interrupted"):
		_write_record(path, [3, 4], {})
	assert path.read_bytes() == before
	assert list(tmp_path.iterdir()) == [path]
	assert _fingerprint({"a": 1, "b": 2}) == _fingerprint({"b": 2, "a": 1})
	assert _fingerprint(np.arange(4)) != _fingerprint(np.arange(4)[::-1])
	assert _cache_path("data", "emb.pkl.gz", {}) != _cache_path(
		"data", "emb.pkl.gz", {"version": 2}
	)
	assert _embedding_settings(
		"wyckoff", {"angle_tolerance": 5.0}
	) == _embedding_settings("wyckoff", {})


def test_audit_comparison(structures):
	"""The full-data audit detects mismatches, dimensions, and failure changes."""
	from xtalmet.distance import _EmbeddingError

	spec = importlib.util.spec_from_file_location(
		"verify_mp20_embeddings",
		Path(__file__).parents[1] / "dev" / "verify_mp20_embeddings.py",
	)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	compare_embedding = module.compare_embedding

	structure = structures[0]
	for metric in ("comp", "wyckoff", "magpie", "amd", "pdd", "elmd", "elmd+amd"):
		assert compare_embedding(metric, (structure, structure.get_embedding(metric)))[
			0
		]
	assert not compare_embedding("comp", (structure, (("Li", 1),)))[0]
	periodic_image = structure.copy()
	periodic_image.translate_sites([0], [1, 0, 0], to_unit_cell=False)
	assert compare_embedding("smat", (structure, periodic_image))[0]
	periodic_image.translate_sites([0], [0.1, 0, 0], to_unit_cell=False)
	assert not compare_embedding("smat", (structure, periodic_image))[0]
	assert not compare_embedding("amd", (structure, np.zeros(2)))[0]
	assert not compare_embedding(
		"amd", (structure, _EmbeddingError(ValueError("failed")))
	)[0]


def test_integer_neighbor_type_is_part_of_identity(structures, tmp_path):
	"""An invalid float neighbor count must not hit a valid integer cache."""
	evaluator = Evaluator(uniqueness=True, distance="amd")
	baseline = evaluator.evaluate(structures, str(tmp_path))[1]
	assert np.any(baseline > 0)
	evaluator.args_emb = {"k": 100.0}
	cached = evaluator.evaluate(structures, str(tmp_path))[1]
	uncached = evaluator.evaluate(structures)[1]
	np.testing.assert_array_equal(cached, uncached)
	np.testing.assert_array_equal(cached, np.zeros(len(structures)))


def test_fingerprint_type_tags_do_not_collide_with_user_dicts(structures, tmp_path):
	"""A malformed dict must not share the fingerprint of a valid float setting."""
	assert _fingerprint(0.01) != _fingerprint({"float": (0.01).hex()})
	evaluator = Evaluator(uniqueness=True, distance="wyckoff")
	evaluator.evaluate(structures, str(tmp_path))
	evaluator.args_emb = {"symprec": {"float": (0.01).hex()}}
	np.testing.assert_array_equal(
		evaluator.evaluate(structures, str(tmp_path))[1],
		evaluator.evaluate(structures)[1],
	)


def test_pdd_row_data_type_does_not_hit_default_cache(structures, tmp_path):
	"""Integer truthy row-data flags retain the backend's distinct return behavior."""
	evaluator = Evaluator(
		uniqueness=True, distance="pdd", args_emb={"k": 4}, args_dist={"n_jobs": 1}
	)
	evaluator.evaluate(structures, str(tmp_path))
	evaluator.args_emb["return_row_data"] = 1
	for directory in (str(tmp_path), None):
		with pytest.raises(ValueError):
			evaluator.evaluate(structures, directory)


def test_prepared_reference_settings_cannot_be_relabelled(structures, tmp_path):
	"""Mutating the original nested kwargs never changes a saved reference's label."""
	args = {"amd": {"k": 4}}
	evaluator = Evaluator(
		novelty=True, distance="elmd+amd", ref_xtals=structures, args_emb=args
	)
	args["amd"]["k"] = 5
	with pytest.raises(ValueError, match="prepared with different"):
		evaluator.evaluate(structures)
	with pytest.raises(ValueError, match="prepared with different"):
		evaluator.save_reference_embeddings(tmp_path / "refs.pkl.gz")
	assert not (tmp_path / "refs.pkl.gz").exists()


def test_magpie_reference_shape_checked_at_load(structures, tmp_path):
	"""Even an internally consistent record must match the real feature count."""
	path = tmp_path / "refs.pkl.gz"
	Evaluator(
		novelty=True, distance="magpie", ref_xtals=structures
	).save_reference_embeddings(path)
	_write_record(path, [[1.0, 2.0]] * len(structures), _read_record(path)["metadata"])
	with pytest.raises(ValueError, match="shape"):
		Evaluator(novelty=True, distance="magpie", ref_xtals=str(path))


class TunableComparator(AbstractComparator):
	"""A valid custom comparator whose inherited as_dict omits its setting."""

	def __init__(self, match):
		"""Keep the comparison flag outside pymatgen's serializer."""
		self.match = match

	def are_equal(self, first, second):
		"""Control whether species match."""
		return self.match

	def get_hash(self, composition):
		"""Keep all compositions as comparison candidates."""
		return 1


def test_custom_comparator_serializer_cannot_hide_settings(tmp_path):
	"""Unsupported custom objects disable matrix reuse instead of colliding."""
	structures = [
		Crystal(np.eye(3) * 3, [element], [[0, 0, 0]]) for element in ("Si", "O")
	]
	evaluator = Evaluator(
		uniqueness=True,
		distance="smat",
		args_dist={"comparator": TunableComparator(True)},
	)
	with pytest.warns(UserWarning, match="Disk cache disabled"):
		first = evaluator.evaluate(structures, str(tmp_path))[1]
	evaluator.args_dist = {"comparator": TunableComparator(False)}
	with pytest.warns(UserWarning, match="Disk cache disabled"):
		second = evaluator.evaluate(structures, str(tmp_path))[1]
	np.testing.assert_array_equal(first, [1, 0])
	np.testing.assert_array_equal(second, [1, 1])
	assert not list(tmp_path.rglob("mtx_uni_smat.pkl.gz"))


def test_phase_patch_order_affects_cache(tmp_path, monkeypatch):
	"""Overlapping patches use the first superset; reordering must invalidate ehull."""
	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.calculator = SimpleNamespace(
		models=[torch.nn.Linear(1, 1)],
		head="default",
		model_type="MACE",
		device="cpu",
		energy_units_to_eV=1,
		length_units_to_A=1,
		get_potential_energy=lambda atoms: 0.0,
	)
	diagram = PatchedPhaseDiagram(
		[PDEntry(element, 0) for element in ("Li", "O", "F", "Cl")]
	)
	for extra, energy in (("F", -2), ("Cl", -4)):
		elements = ("Li", "O", extra)
		diagram[frozenset(Element(element) for element in elements)] = PhaseDiagram(
			[PDEntry(element, 0) for element in elements] + [PDEntry("LiO", energy)]
		)
	calculator.ppd_mp = diagram
	calculator.binary, calculator.threshold, calculator.intercept = False, 0.1, 3.0
	monkeypatch.setattr("xtalmet.evaluator.StabilityCalculator", lambda **k: calculator)
	evaluator = Evaluator(stability="continuous")
	structure = Crystal(np.eye(3) * 4, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
	first = evaluator.evaluate([structure], str(tmp_path))[1]
	diagram.pds = dict(reversed(list(diagram.pds.items())))
	second = evaluator.evaluate([structure], str(tmp_path))[1]
	np.testing.assert_allclose(first, [2 / 3])
	np.testing.assert_allclose(second, [1 / 3])
	np.testing.assert_allclose(second, evaluator.evaluate([structure])[1])


def test_custom_phase_behavior_disables_reuse(structures, tmp_path, monkeypatch):
	"""Extra subclass behavior cannot collide merely because phase entries match."""

	class ShiftedDiagram(PatchedPhaseDiagram):
		"""A supported input whose custom hull calculation uses extra state."""

		shift = 0.0

		def get_e_above_hull(self, entry, **kwargs):
			"""Return a custom energy without changing serialized phase entries."""
			return self.shift

	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.ppd_mp = ShiftedDiagram([PDEntry("Si", 0)])
	calculator.calculator = SimpleNamespace(get_potential_energy=lambda atoms: 0.0)
	calculator.binary, calculator.threshold, calculator.intercept = False, 0.1, 3.0
	monkeypatch.setattr("xtalmet.evaluator.StabilityCalculator", lambda **k: calculator)
	evaluator = Evaluator(stability="continuous")
	with pytest.warns(UserWarning, match="custom phase-diagram"):
		first = evaluator.evaluate(structures[:1], str(tmp_path))[1]
	calculator.ppd_mp.shift = 1.0
	with pytest.warns(UserWarning, match="custom phase-diagram"):
		second = evaluator.evaluate(structures[:1], str(tmp_path))[1]
	np.testing.assert_allclose(first, [1.0])
	np.testing.assert_allclose(second, [2 / 3])
	assert not list(tmp_path.rglob("ehull.pkl.gz"))


@pytest.mark.parametrize("nested", [False, True])
def test_custom_model_state_is_not_assumed_complete(nested):
	"""A custom forward method may depend on state absent from state_dict/repr."""

	class CustomModel(torch.nn.Linear):
		"""An unknown model can use extra non-tensor state."""

	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.ppd_mp = PatchedPhaseDiagram([PDEntry("Si", 0)])
	model = CustomModel(1, 1)
	if nested:
		model = torch.nn.Sequential(model)
	calculator.calculator = SimpleNamespace(models=[model])
	with pytest.raises(TypeError, match="custom model"):
		calculator._cache_identity()


def test_unserializable_model_disables_stability_cache(
	structures, tmp_path, monkeypatch
):
	"""An unsupported standalone ScriptModule must not prevent evaluation."""
	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.calculator = SimpleNamespace(
		models=[torch.jit.script(torch.nn.Linear(1, 1))]
	)
	calculator.ppd_mp = PatchedPhaseDiagram([PDEntry("Si", 0)])
	calculator.binary, calculator.threshold = True, 0.1
	calculator._ehull = lambda xtals: np.zeros(len(xtals))
	monkeypatch.setattr("xtalmet.evaluator.StabilityCalculator", lambda **k: calculator)
	evaluator = Evaluator(stability="binary")
	with pytest.warns(UserWarning, match="model cannot be serialized"):
		np.testing.assert_array_equal(
			evaluator.evaluate(structures, str(tmp_path))[1], np.ones(len(structures))
		)
	assert not list(tmp_path.rglob("ehull.pkl.gz"))
