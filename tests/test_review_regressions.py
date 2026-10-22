"""Regressions for nonfinite structures, energies, distances and settings."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry

from xtalmet.crystal import Crystal, _magpie_featurizer
from xtalmet.distance import _EmbeddingError, distance, distance_matrix
from xtalmet.evaluator import Evaluator
from xtalmet.stability import StabilityCalculator

distance_module = importlib.import_module("xtalmet.distance")


def crystal():
	"""Make a finite, ordered structure with two sites."""
	return Crystal(np.eye(3) * 3, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["coords", "lattice"])
@pytest.mark.parametrize("parallel", [False, True])
def test_smat_nonfinite_geometry(bad, field, parallel, monkeypatch, tmp_path):
	"""Reject bad geometry before native matching, preserving both matrix forms."""
	good = crystal()
	lattice = np.eye(3) * 3
	coords = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
	if field == "coords":
		coords[1, 0] = bad
	else:
		lattice[0, 0] = bad
	with np.errstate(invalid="ignore"):
		invalid = Crystal(lattice, ["Si", "Si"], coords)
	original = distance_module._DIST_MTX_FUNCS["smat"]

	def checked_backend(left, right, *args, **kwargs):
		for item in left + (right or []):
			assert np.isfinite(item.cart_coords).all()
		return original(left, right, *args, **kwargs)

	# Fail with an assertion rather than crashing pytest if the guard regresses.
	monkeypatch.setitem(distance_module._DIST_MTX_FUNCS, "smat", checked_backend)
	monkeypatch.setitem(
		distance_module._DIST_FUNCS,
		"smat",
		lambda *a, **k: pytest.fail("invalid pair reached matcher"),
	)
	with np.errstate(invalid="ignore"):
		assert np.isnan(distance("smat", good, invalid))
		for right in (None, [invalid, good]):
			actual = distance_matrix(
				"smat",
				[good, invalid, good],
				right,
				multiprocessing=parallel,
				n_processes=2,
			)
			expected = (
				[[0, np.nan, 0], [np.nan, np.nan, np.nan], [0, np.nan, 0]]
				if right is None
				else [[np.nan, 0], [np.nan, np.nan], [np.nan, 0]]
			)
			np.testing.assert_allclose(actual, expected, equal_nan=True)
		evaluator = Evaluator(
			validity=["structure"],
			uniqueness=True,
			novelty=True,
			distance="smat",
			ref_xtals=[invalid, good],
			agg_func="ave",
			weights={"validity": 0, "uniqueness": 1, "novelty": 1},
		)
		for _ in range(2):
			_, scores, _ = evaluator.evaluate(
				[good, invalid, good], str(tmp_path), parallel, 2
			)
			np.testing.assert_array_equal(scores, [0.5, 0, 0])


@pytest.fixture
def stability():
	"""Use a small real hull with a deterministic energy calculator."""
	calculator = StabilityCalculator.__new__(StabilityCalculator)
	calculator.binary = True
	calculator.threshold = 0.1
	calculator.intercept = 0.4289
	calculator.ppd_mp = PatchedPhaseDiagram(
		[PDEntry("Zn", -1), PDEntry("O2", -2), PDEntry("ZnO", -4)]
	)
	calculator.calculator = SimpleNamespace(get_potential_energy=lambda atoms: -8.0)
	return calculator


@pytest.mark.parametrize("binary", [False, True])
def test_stability_precomputed_nonfinite(stability, binary):
	"""Only finite energies may receive positive stability, including negative ones."""
	stability.binary = binary
	values = np.array([np.nan, np.inf, -np.inf, -0.2, 0.0, 0.05, 0.2])
	before = values.copy()
	scores, raw, _ = stability.compute_stability_scores(
		[crystal()] * len(values), values
	)
	expected = (
		[0, 0, 0, 1, 1, 1, 0]
		if binary
		else [
			0,
			0,
			0,
			1,
			1,
			1 - 0.05 / stability.intercept,
			1 - 0.2 / stability.intercept,
		]
	)
	np.testing.assert_allclose(scores, expected)
	np.testing.assert_array_equal(raw, before)
	np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("source", ["energy", "hull"])
def test_stability_computed_nonfinite(stability, bad, source):
	"""Nonfinite model or hull output remains a failed energy with score zero."""
	if source == "energy":
		stability.calculator.get_potential_energy = lambda atoms: bad
		stability.ppd_mp = SimpleNamespace(
			get_e_above_hull=lambda *a, **k: pytest.fail(
				"nonfinite energy reached hull"
			)
		)
	else:
		stability.ppd_mp = SimpleNamespace(get_e_above_hull=lambda *a, **k: bad)
	for binary in (False, True):
		stability.binary = binary
		scores, raw, _ = stability.compute_stability_scores([crystal()])
		np.testing.assert_array_equal(scores, [0])
		assert np.isnan(raw).all()


def test_stability_oxidation_states_preserved(stability):
	"""Use elemental composition for the hull without mutating decorated sites."""
	decorated = Crystal.from_file("tests/data/wz-ZnO.cif")
	before = decorated.as_dict()
	neutral = Crystal.from_Structure(decorated)
	assert decorated.composition != neutral.composition
	actual = stability.compute_stability_scores([decorated, neutral])
	np.testing.assert_array_equal(actual[0], [1, 1])
	np.testing.assert_allclose(actual[1], [0, 0], atol=1e-14)
	assert decorated.as_dict() == before


def test_stability_invalid_coordinates_skip_model(stability):
	"""Keep nonfinite coordinates out of native model neighbor searches."""
	invalid = Crystal(np.eye(3) * 3, ["Si"], [[np.nan, 0, 0]])
	stability.calculator.get_potential_energy = lambda atoms: pytest.fail(
		"model called"
	)
	assert np.isnan(stability._ehull([invalid])).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("disk", [False, True])
def test_nonfinite_pair_aggregation(bad, disk, monkeypatch, tmp_path):
	"""U keeps its denominator; N uses finite pairs and zero for an empty row."""

	def backend(left, right, *args, **kwargs):
		return np.array(
			[[0, bad, 3], [bad, 0, bad], [3, bad, 0]]
			if right is None
			else [[bad, 1], [bad, bad], [3, bad]],
			dtype=float,
		)

	monkeypatch.setitem(distance_module._DIST_MTX_FUNCS, "amd", backend)
	monkeypatch.setattr(Crystal, "get_embedding", lambda *a, **k: np.ones(100))
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance="amd",
		ref_xtals=[crystal()] * 2,
		agg_func="ave",
	)
	for _ in range(2):
		_, scores, _ = evaluator.evaluate(
			[crystal()] * 3, str(tmp_path) if disk else None
		)
		np.testing.assert_allclose(scores, [(0.375 + 0.5) / 2, 0, (0.375 + 0.75) / 2])


def test_undefined_correlation_scores_zero():
	"""A supported metric can yield NaN despite finite, real AMD embeddings."""
	xs = [Crystal(np.eye(3) * size, ["Si"], [[0, 0, 0]]) for size in (3, 4)]
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance="amd",
		ref_xtals=xs,
		args_emb={"k": 1},
		args_dist={"metric": "correlation"},
	)
	np.testing.assert_array_equal(evaluator.evaluate(xs)[1], [0, 0])


@pytest.mark.parametrize("normalize", [False, True])
def test_overflow_distance_preserves_raw_out(normalize):
	"""Represent backend overflow as NaN without replacing raw output with NaN."""
	a, b = [1e200, 1e200], [2e200, 2e200]
	out = np.empty((1, 1))
	args = {"metric": "euclidean", "out": out}
	assert np.isnan(
		distance("amd", a, b, normalize=normalize, args_dist={"metric": "euclidean"})
	)
	actual = distance_matrix("amd", [a], [b], normalize=normalize, args_dist=args)
	assert np.isnan(actual).all()
	assert np.isposinf(out).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0])
@pytest.mark.parametrize("agg", ["ave", "prod"])
def test_invalid_weights(bad, agg):
	"""Reject invalid coefficients before evaluating or preparing data."""
	with pytest.raises(ValueError, match="finite.*non-negative"):
		Evaluator(validity=["structure"], agg_func=agg, weights={"validity": bad})


def test_weight_sum_overflow():
	"""Finite individual weights must not produce a nonfinite enabled sum."""
	with pytest.raises(ValueError, match="finite positive"):
		Evaluator(
			validity=[],
			uniqueness=True,
			distance="comp",
			agg_func="ave",
			weights={"validity": 1e308, "uniqueness": 1e308},
		)


@pytest.mark.parametrize(
	"coefs",
	[
		{"elmd": 0, "amd": 0},
		{"elmd": -1, "amd": 2},
		{"elmd": np.nan, "amd": 1},
		{"elmd": 1, "amd": np.inf},
		{"elmd": 1, "amd": -np.inf},
		{"elmd": 1e308, "amd": 1e308},
	],
)
def test_invalid_blending_coefficients(coefs):
	"""Validate coefficients even when all inputs fail before blending."""
	error = _EmbeddingError(ValueError("failed"))
	args = {"args_dist": {"coefs": coefs}}
	with pytest.raises(ValueError, match="coefficients"):
		distance("elmd+amd", error, error, **args)
	with pytest.raises(ValueError, match="coefficients"):
		distance_matrix("elmd+amd", [error], **args)
	with pytest.raises(ValueError, match="coefficients"):
		Evaluator(uniqueness=True, distance="elmd+amd", **args)


@pytest.mark.parametrize(
	"binary, value",
	[
		(True, np.nan),
		(True, np.inf),
		(True, -np.inf),
		(True, None),
		(False, np.nan),
		(False, np.inf),
		(False, -np.inf),
		(False, 0),
		(False, -1),
		(False, None),
	],
)
def test_invalid_stability_settings(binary, value, monkeypatch):
	"""Reject invalid active settings before downloads or model initialization."""
	monkeypatch.setattr(
		"xtalmet.stability.hf_hub_download", lambda **k: pytest.fail("download")
	)
	kwargs = {"threshold": value} if binary else {"intercept": value}
	with pytest.raises(ValueError, match="finite"):
		StabilityCalculator(binary=binary, **kwargs)
	monkeypatch.setattr(
		Evaluator,
		"_prepare_ref_xtals",
		lambda *a, **k: pytest.fail("references prepared"),
	)
	with pytest.raises(ValueError, match="finite"):
		Evaluator(
			stability="binary" if binary else "continuous",
			novelty=True,
			distance="amd",
			ref_xtals=[crystal()],
			args_stability=kwargs,
		)


def test_magpie_default_reuse(monkeypatch):
	"""Load the default tables once while keeping custom instance featurizers."""
	from matminer.featurizers.composition.composite import ElementProperty

	expected = ElementProperty.from_preset("magpie", impute_nan=True)
	xs = [crystal(), Crystal(np.eye(3) * 3, ["Zn", "O"], [[0, 0, 0], [0.5] * 3])]
	values = [expected.featurize(x.composition) for x in xs]
	calls = []
	_magpie_featurizer.cache_clear()
	monkeypatch.setattr(
		ElementProperty, "from_preset", lambda *a, **k: calls.append(1) or expected
	)
	try:
		actual = [x.get_embedding("magpie") for x in xs]
		np.testing.assert_array_equal(actual, values)
		assert len(calls) == 1
		assert all(not hasattr(x, "featurizer") for x in xs)
		custom = crystal()
		custom.featurizer = SimpleNamespace(featurize=lambda comp: [42.0])
		assert custom.get_embedding("magpie") == [42.0]
		assert len(calls) == 1
	finally:
		_magpie_featurizer.cache_clear()
