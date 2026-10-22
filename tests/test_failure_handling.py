"""Regression tests for failed embeddings and evaluation boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from xtalmet._cache import (
	_canonical,
	_embedding_settings,
	_read_record,
	_write_record,
)
from xtalmet.crystal import Crystal
from xtalmet.distance import (
	_compute_embeddings,
	_EmbeddingError,
	distance,
	distance_matrix,
)
from xtalmet.evaluator import Evaluator

ERROR = _EmbeddingError(RuntimeError("embedding failed"))
METRICS = ("smat", "comp", "wyckoff", "magpie", "pdd", "amd", "elmd", "elmd+amd")


def embedding_args(metric):
	"""Declare the two-neighbor dimensions used by these synthetic embeddings."""
	if metric in ("amd", "pdd"):
		return {"k": 2}
	return {"amd": {"k": 2}} if metric == "elmd+amd" else {}


def write_reference(path, metric, values):
	"""Write a versioned reference fixture, including synthetic dimensions."""
	_write_record(
		path,
		values,
		{
			"kind": "reference_embeddings",
			"distance": metric,
			"args_emb": _canonical(_embedding_settings(metric, embedding_args(metric))),
			"count": len(values),
		},
	)


def crystals(n):
	"""Give each structure an identifiable lattice for mocked embedding failures."""
	return [Crystal(np.eye(3) * (3 + i), ["Si"], [[0, 0, 0]]) for i in range(n)]


def embeddings(metric):
	"""Small valid inputs for every distance backend."""
	return {
		"smat": crystals(2),
		"comp": [(("Si", 1),), (("O", 1),)],
		"wyckoff": [(1, ("a",)), (2, ("b",))],
		"magpie": [[1.0, 2.0] + [0.0] * 130, [3.0, 4.0] + [0.0] * 130],
		"pdd": [np.array([[1.0, 1.0, 2.0]]), np.array([[1.0, 3.0, 4.0]])],
		"amd": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
		"elmd": ["Si", "O"],
		"elmd+amd": [("Si", np.array([1.0, 2.0])), ("O", np.array([3.0, 4.0]))],
	}[metric]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_failed_distance_inputs(metric, two_sets, parallel):
	"""Failures retain their positions, including diagonals, in both matrix forms."""
	a, b = embeddings(metric)
	left = [ERROR, a, ERROR, b]
	right = [b, ERROR, a] if two_sets else None
	args = {"n_jobs": 2 if parallel else 1} if metric == "pdd" else {}
	actual = distance_matrix(
		metric,
		left,
		right,
		multiprocessing=parallel,
		n_processes=2,
		args_dist=args,
		verbose=True,
	)
	target = right if two_sets else left
	expected = np.full((len(left), len(target)), np.nan)
	for i, x in enumerate(left):
		for j, y in enumerate(target):
			if x is not ERROR and y is not ERROR:
				expected[i, j] = distance(metric, x, y, args_dist=args)
	np.testing.assert_allclose(actual[0], expected, equal_nan=True, atol=1e-13)
	assert actual[1][0] is ERROR and actual[1][2] is ERROR
	assert np.isnan(distance(metric, ERROR, a))
	assert np.isnan(distance(metric, a, ERROR))
	assert np.isnan(distance(metric, ERROR, ERROR))
	for first, second, shape in (
		([ERROR, ERROR], None, (2, 2)),
		([ERROR], [a, b], (1, 2)),
		([a, b], [ERROR], (2, 1)),
	):
		matrix = distance_matrix(metric, first, second, args_dist=args)
		assert matrix.shape == shape
		assert np.isnan(matrix).all()
	np.testing.assert_allclose(distance_matrix(metric, [a]), [[0.0]], atol=1e-13)
	one_valid = distance_matrix(metric, [ERROR, a, ERROR], args_dist=args)
	expected_one = np.full((3, 3), np.nan)
	expected_one[1, 1] = 0
	np.testing.assert_allclose(one_valid, expected_one, atol=1e-13)


@pytest.mark.parametrize("metric", ["magpie", "amd", "pdd", "elmd+amd"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_embedding(metric, bad):
	"""Reject the whole embedding, even for raw distances or a zero blend weight."""
	a, _ = embeddings(metric)
	invalid = ("Si", [bad, 2.0]) if metric == "elmd+amd" else np.full_like(a, bad)
	for normalize in (False, True):
		assert np.isnan(distance(metric, invalid, a, normalize=normalize))
		matrix = distance_matrix(metric, [a, invalid], normalize=normalize)
		np.testing.assert_allclose(matrix, [[0.0, np.nan], [np.nan, np.nan]])
	if metric == "elmd+amd":
		assert np.isnan(
			distance(
				metric,
				invalid,
				a,
				args_dist={
					"coefs": {"elmd": 1, "amd": 0},
				},
			)
		)


def test_blend_component_failure():
	"""A failed blend component never falls back to its surviving component."""
	a = embeddings("elmd+amd")[0]
	for failed in ((ERROR, a[1]), (a[0], ERROR)):
		assert np.isnan(distance("elmd+amd", failed, a))
		np.testing.assert_allclose(
			distance_matrix("elmd+amd", [a, failed]),
			[[0, np.nan], [np.nan, np.nan]],
		)


def patch_embeddings(monkeypatch, values):
	"""Exercise the real embedding worker, including its exception sentinel."""

	def get_embedding(self, metric, **kwargs):
		value = values[round(self.lattice.a) - 3]
		if isinstance(value, _EmbeddingError):
			raise value.exc
		return value

	monkeypatch.setattr(Crystal, "get_embedding", get_embedding)


@pytest.mark.parametrize(
	"metric", ["comp", "wyckoff", "magpie", "amd", "pdd", "elmd", "elmd+amd"]
)
@pytest.mark.parametrize("cache_mode", ["none", "complete", "matrix_only"])
def test_evaluator_failure_masks(metric, cache_mode, tmp_path, monkeypatch):
	"""Cache hits use embedding masks, regardless of failures in the first row."""
	a, b = embeddings(metric)
	values = [ERROR, a, ERROR, b, a]
	patch_embeddings(monkeypatch, values)
	ref_path = tmp_path / "refs.pkl.gz"
	write_reference(ref_path, metric, [ERROR, a])
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance=metric,
		ref_xtals=str(ref_path),
		agg_func="ave",
		weights={"uniqueness": 1, "novelty": 2},
		args_emb=embedding_args(metric),
		args_dist={"n_jobs": 1} if metric == "pdd" else {},
	)
	u = np.array([0, 1, 0, 1, 0], dtype=float)
	n = np.array([0, 0, 0, 1, 0], dtype=float)
	if metric not in ("comp", "wyckoff"):
		d = distance(metric, a, b)
		u = np.array([0, d / 4, 0, d / 2, d / 4])
		n = np.array([0, 0, 0, d, 0])
	cache = str(tmp_path / "cache") if cache_mode != "none" else None
	_, scores, _ = evaluator.evaluate(crystals(5), dir_intermediate=cache)
	np.testing.assert_allclose(scores, (u + 2 * n) / 3)
	if cache is not None:
		matrix_path = next((tmp_path / "cache").rglob(f"mtx_uni_{metric}.pkl.gz"))
		embedding_path = next((tmp_path / "cache").rglob(f"emb_{metric}.pkl.gz"))
		matrix_bytes = matrix_path.read_bytes()
		if cache_mode == "matrix_only":
			embedding_path.unlink()
		else:
			monkeypatch.setattr(
				Crystal, "get_embedding", lambda *a, **k: pytest.fail("cache miss")
			)
		_, cached, times = evaluator.evaluate(crystals(5), dir_intermediate=cache)
		np.testing.assert_allclose(cached, scores)
		assert matrix_path.read_bytes() == matrix_bytes
		assert times["uni_d_mtx"] == times["nov_d_mtx"] == 0
		assert embedding_path.is_file()
		assert (times["uni_emb"] > 0) == (cache_mode == "matrix_only")
	else:
		for value in (ERROR, a):
			patch_embeddings(monkeypatch, [value] * 5)
			_, repeated, _ = evaluator.evaluate(crystals(5))
			assert np.isfinite(repeated).all()
			if value is ERROR:
				np.testing.assert_array_equal(repeated, np.zeros(5))


@pytest.mark.parametrize("agg", ["ave", "prod"])
@pytest.mark.parametrize("weight", [0, 1])
def test_embedding_failure_does_not_change_validity_or_stability(
	monkeypatch, agg, weight
):
	"""Failed U/N never overwrites V/S or forces the final weighted score to zero."""
	patch_embeddings(monkeypatch, [ERROR, [1.0, 2.0]])
	monkeypatch.setattr(
		"xtalmet.evaluator.StabilityCalculator", lambda **k: SimpleNamespace()
	)
	evaluator = Evaluator(
		validity=[],
		stability="continuous",
		uniqueness=True,
		distance="magpie",
		agg_func=agg,
		weights={"validity": 1, "stability": 1, "uniqueness": weight},
	)
	monkeypatch.setattr(
		evaluator, "_evaluate_stability", lambda *a: (np.array([0.5, 1.0]), {})
	)
	_, scores, _ = evaluator.evaluate(crystals(2))
	expected = np.array([1.5, 2.0]) / (2 + weight) if agg == "ave" else [0.5, 1.0]
	if agg == "prod" and weight:
		expected = [0, 0]
	np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize("metric", METRICS)
def test_empty_distance_sets(metric):
	"""Reject either explicit empty input before indexing or invoking a backend."""
	a, _ = embeddings(metric)
	for left, right in (([], None), ([], [a]), ([a], [])):
		with pytest.raises(ValueError, match="empty|at least one"):
			distance_matrix(metric, left, right)


def test_evaluator_boundaries(tmp_path, monkeypatch):
	"""Reject undefined scores before model work or creating cache directories."""
	evaluator = Evaluator(uniqueness=True, distance="magpie")
	for samples in ([], crystals(1)):
		with pytest.raises(ValueError, match="at least|empty"):
			evaluator.evaluate(samples, str(tmp_path / "unused"))
	assert not (tmp_path / "unused").exists()
	with pytest.raises(ValueError, match="reference|Reference"):
		Evaluator(novelty=True, distance="comp", ref_xtals=[])
	for refs in ([], [ERROR, ERROR]):
		path = tmp_path / "refs.pkl.gz"
		Evaluator._write_pickle_gz(None, refs, str(path))
		with pytest.raises(ValueError, match="reference|Reference"):
			Evaluator(novelty=True, distance="comp", ref_xtals=str(path))
	monkeypatch.setattr(
		"xtalmet.evaluator.StabilityCalculator", lambda **k: pytest.fail("model loaded")
	)
	with pytest.raises(ValueError, match="weight"):
		Evaluator(
			stability="continuous",
			agg_func="ave",
			weights={"stability": 0, "validity": 1},
		)
	assert Evaluator(uniqueness=True, distance="comp").evaluate(crystals(1))[1][0] == 1
	assert (
		Evaluator(novelty=True, distance="comp", ref_xtals=crystals(1)).evaluate(
			crystals(1)
		)[1][0]
		== 0
	)
	assert Evaluator(validity=[]).evaluate(crystals(1))[1][0] == 1
	with pytest.raises(ValueError, match="at least one"):
		Evaluator(validity=[]).evaluate([])
	assert (
		Evaluator(novelty=True, distance="magpie", ref_xtals=crystals(1)).evaluate(
			crystals(1)
		)[1][0]
		== 0
	)


@pytest.mark.parametrize("metric", ["amd", "elmd+amd"])
@pytest.mark.parametrize("two_sets", [False, True])
@pytest.mark.parametrize("all_failed", [False, True])
def test_failed_amd_output_buffer(metric, two_sets, all_failed):
	"""Preserve raw AMD output in full rectangular or condensed input positions."""
	a, b = embeddings(metric)
	left = [ERROR, a, b, ERROR, a] if not all_failed else [ERROR] * 5
	right = [b, ERROR, a] if two_sets else None
	target = right if two_sets else left
	raw = np.full((len(left), len(target)), np.nan)
	for i, x in enumerate(left):
		for j, y in enumerate(target):
			if x is not ERROR and y is not ERROR:
				x_amd, y_amd = (x[1], y[1]) if metric == "elmd+amd" else (x, y)
				raw[i, j] = distance("amd", x_amd, y_amd, normalize=False)
	expected_out = raw if two_sets else raw[np.triu_indices(len(left), 1)]
	out = np.empty(expected_out.shape)
	args = {"out": out} if metric == "amd" else {"amd": {"out": out}}
	actual = distance_matrix(metric, left, right, args_dist=args)
	np.testing.assert_allclose(out, expected_out, equal_nan=True)
	assert not np.shares_memory(actual, out)
	expected = distance_matrix(metric, left, right)
	np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_failed_amd_readonly_output():
	"""Filtering must retain the backend's requirement for writable output."""
	out = np.ones((2, 1))
	out.flags.writeable = False
	with pytest.raises(ValueError, match="writable"):
		distance_matrix(
			"amd", [ERROR, [1.0, 2.0]], [[3.0, 4.0]], args_dist={"out": out}
		)
	np.testing.assert_array_equal(out, np.ones((2, 1)))


@pytest.mark.parametrize("metric", ["magpie", "amd", "pdd", "elmd+amd"])
def test_nonfinite_evaluator_masks(metric, tmp_path, monkeypatch):
	"""Numerical failures affect U/N masks for generated and reference inputs."""
	a, b = embeddings(metric)
	invalid = ("Si", [np.inf, 2.0]) if metric == "elmd+amd" else np.full_like(a, np.inf)
	patch_embeddings(monkeypatch, [invalid, a, b])
	path = tmp_path / "refs.pkl.gz"
	write_reference(path, metric, [invalid, a])
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance=metric,
		ref_xtals=str(path),
		agg_func="ave",
		args_emb=embedding_args(metric),
		args_dist={"n_jobs": 1} if metric == "pdd" else {},
	)
	d = distance(metric, a, b)
	_, scores, _ = evaluator.evaluate(crystals(3))
	np.testing.assert_allclose(scores, [0, d / 4, 3 * d / 4])
	write_reference(path, metric, [invalid])
	with pytest.raises(ValueError, match="valid reference"):
		Evaluator(
			novelty=True,
			distance=metric,
			ref_xtals=str(path),
			args_emb=embedding_args(metric),
		)


def test_smat_evaluator_failed_cache(tmp_path):
	"""Binary StructureMatcher scores also use explicit cached embedding masks."""
	a = crystals(1)[0]
	b = Crystal(np.eye(3) * 3, ["O"], [[0, 0, 0]])
	write_reference(tmp_path / "refs.pkl.gz", "smat", [ERROR, a])
	evaluator = Evaluator(
		uniqueness=True,
		novelty=True,
		distance="smat",
		ref_xtals=str(tmp_path / "refs.pkl.gz"),
		agg_func="ave",
	)
	evaluator.evaluate(crystals(4), str(tmp_path))
	path = next(tmp_path.rglob("emb_smat.pkl.gz"))
	_write_record(path, [ERROR, a, b, a], _read_record(path)["metadata"])
	for _ in range(2):
		_, scores, _ = evaluator.evaluate(crystals(4), str(tmp_path))
		np.testing.assert_array_equal(scores, [0, 0.5, 1, 0])


@pytest.mark.parametrize("parallel", [False, True])
def test_embedding_worker_failures(parallel):
	"""Real worker exceptions become picklable failures with serial or two workers."""
	embs = _compute_embeddings("unsupported", crystals(2), parallel, n_processes=2)
	assert all(isinstance(emb, _EmbeddingError) for emb in embs)
	assert np.isnan(distance_matrix("comp", embs)).all()
