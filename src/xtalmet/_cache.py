"""Content identities and versioned records for evaluator caches."""

import gzip
import hashlib
import inspect
import json
import os
import pickle
import sys
import tempfile
import warnings
import zlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import amd
import numpy as np

from .crystal import Crystal
from .distance import _EmbeddingError

CACHE_FORMAT_VERSION = 1
MP20_REFERENCE_REVISION = "b1a59e5e0fa0e348e5907494e4cf8f16147b3932"


def _canonical(value: Any) -> Any:
	"""Represent supported values deterministically, without copying dense arrays."""
	if value is None or isinstance(value, (str, bool, int)):
		return value
	if isinstance(value, float):
		return {"float": value.hex()}
	if isinstance(value, np.generic):
		return _canonical(value.item())
	if isinstance(value, np.ndarray):
		if value.dtype.hasobject:
			raise TypeError("object arrays cannot be fingerprinted")
		return {
			"array": hashlib.sha256(np.ascontiguousarray(value)).hexdigest(),
			"dtype": value.dtype.str,
			"shape": list(value.shape),
		}
	if isinstance(value, _EmbeddingError):
		return {"embedding_error": [type(value.exc).__qualname__, str(value.exc)]}
	if isinstance(value, dict) and all(isinstance(key, str) for key in value):
		return {
			"dict": [[key, _canonical(item)] for key, item in sorted(value.items())]
		}
	if isinstance(value, (list, tuple)):
		return {type(value).__name__: [_canonical(item) for item in value]}
	if isinstance(value, Path):
		return {"path": str(value)}
	if (
		type(value).__module__.startswith("pymatgen.") and hasattr(value, "as_dict")
	) or isinstance(value, Crystal):
		return {
			"class": f"{type(value).__module__}.{type(value).__qualname__}",
			"state": _canonical(value.as_dict()),
		}
	raise TypeError(f"cannot fingerprint {type(value).__qualname__}")


def _fingerprint(value: Any) -> str:
	"""Hash supported values without relying on object identity or repr."""
	encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(encoded.encode()).hexdigest()


def _embedding_settings(distance: str, args: dict) -> dict:
	"""Resolve the settings actually used by Crystal.get_embedding."""
	if distance == "elmd+amd":
		return {"amd": _embedding_settings("amd", args.get("amd", {}))}
	method = getattr(Crystal, f"_get_emb_d_{distance}", None)
	if distance in ("smat", "comp", "magpie", "elmd"):
		return {}
	settings = {
		name: param.default
		for name, param in inspect.signature(method).parameters.items()
		if name != "self" and param.default is not inspect.Parameter.empty
	}
	if distance == "pdd":
		settings.update(
			{
				name: param.default
				for name, param in inspect.signature(amd.PDD).parameters.items()
				if param.default is not inspect.Parameter.empty
			}
		)
	defaults = settings.copy()
	settings.update(args)
	for name, default in defaults.items():
		if (
			isinstance(settings[name], (int, float, np.number))
			and (name != "k" or isinstance(settings[name], (int, np.integer)))
			and settings[name] == default
		):
			settings[name] = default
	if distance == "pdd" and settings["return_row_data"] is True:
		settings["return_row_data"] = False
	return settings


def _software_identity() -> dict:
	"""Identify runtime dependencies and local changes even before a release bump."""
	packages = {}
	for name in (
		"xtalmet",
		"numpy",
		"scipy",
		"pymatgen",
		"pymatgen-core",
		"spglib",
		"average-minimum-distance",
		"ElMD",
		"matminer",
		"smact",
		"mace-torch",
		"torch",
		"e3nn",
		"ase",
		"numba",
	):
		try:
			packages[name] = version(name)
		except PackageNotFoundError:
			packages[name] = None
	return {
		"python": list(sys.version_info[:3]),
		"packages": packages,
		"implementation": {
			name: hashlib.sha256(
				Path(__file__).with_name(name).read_bytes()
			).hexdigest()
			for name in (
				"_cache.py",
				"evaluator.py",
				"crystal.py",
				"distance.py",
				"validity.py",
				"stability.py",
				"constants.py",
			)
		},
	}


def _metadata(context: dict | None, **settings: Any) -> dict | None:
	"""Disable only the affected disk cache when settings cannot be identified."""
	if context is None:
		return None
	try:
		return {
			key: _canonical(value) for key, value in {**context, **settings}.items()
		}
	except TypeError as exc:
		warnings.warn(f"Disk cache disabled: {exc}.", UserWarning, stacklevel=2)
		return None


def _cache_path(directory: str, filename: str, metadata: dict) -> Path:
	"""Locate an artifact by format and calculation identity."""
	return (
		Path(directory)
		/ "cache"
		/ f"format-{CACHE_FORMAT_VERSION}"
		/ _fingerprint(metadata)
		/ filename
	)


def _write_record(path: str | Path, data: Any, metadata: dict) -> None:
	"""Commit a complete gzip-pickle record by atomic replacement."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	record = {
		"cache_format_version": CACHE_FORMAT_VERSION,
		"metadata": metadata,
		"data_fingerprint": _fingerprint(data),
		"data": data,
	}
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
			temporary = Path(handle.name)
			with gzip.GzipFile(fileobj=handle, mode="wb") as compressed:
				pickle.dump(record, compressed, protocol=pickle.HIGHEST_PROTOCOL)
		os.replace(temporary, path)
	finally:
		if temporary is not None:
			temporary.unlink(missing_ok=True)


def _read_record(path: str | Path) -> dict:
	"""Read a record and reject unknown formats or inconsistent payloads."""
	try:
		with gzip.open(path, "rb") as handle:
			record = pickle.load(handle)
	except zlib.error as exc:
		raise ValueError(f"Invalid compressed record: {path}") from exc
	if (
		not isinstance(record, dict)
		or record.get("cache_format_version") != CACHE_FORMAT_VERSION
		or not isinstance(record.get("metadata"), dict)
		or "data" not in record
		or record.get("data_fingerprint") != _fingerprint(record["data"])
	):
		raise ValueError(f"Invalid versioned cache/reference record: {path}")
	return record
