"""Build committed main/tag documentation without changing the source checkout.

Run ``uv run --locked python docs/build_docs.py`` from the repository root, or
``uv run --locked python build_docs.py`` from docs. All versions must build before
HTML is copied to docs/build/html; unrelated files in that directory are retained.
"""

import os
import shutil
import subprocess
import sys
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_ref(ref: str) -> str:
	"""Resolve a local Git reference to a commit without checking it out."""
	return subprocess.check_output(
		["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
		cwd=PROJECT_ROOT,
		text=True,
	).strip()


def checkout(destination: Path, commit: str) -> None:
	"""Create a detached checkout in a temporary local clone."""
	subprocess.run(
		[
			"git",
			"clone",
			"--shared",
			"--no-checkout",
			"--quiet",
			str(PROJECT_ROOT),
			str(destination),
		],
		check=True,
	)
	subprocess.run(
		["git", "checkout", "--detach", "--quiet", commit],
		cwd=destination,
		check=True,
	)


def build_doc(version: str, checkout_path: Path) -> Path:
	"""Generate API pages and HTML using only this checkout's package source."""
	env = os.environ.copy()
	env.update(
		BUILD_ALL_DOCS="True",
		PAGES_ROOT="https://wmd-group.github.io/xtalmet",
		CURRENT_VERSION=version,
		PYTHONPATH=str(checkout_path / "src") + os.pathsep + env.get("PYTHONPATH", ""),
	)
	docs_path = checkout_path / "docs"
	api_path = docs_path / "source/api"
	if api_path.exists():
		shutil.rmtree(api_path)
	subprocess.run(
		[
			sys.executable,
			"-m",
			"sphinx.ext.apidoc",
			"-o",
			"source/api",
			"-f",
			"-T",
			"--remove-old",
			"-e",
			"-M",
			"-t",
			"source/_templates/apidoc",
			"../src/xtalmet",
		],
		cwd=docs_path,
		env=env,
		check=True,
	)
	subprocess.run(
		[sys.executable, "-m", "sphinx", "-M", "html", "source", "_build_tag"],
		cwd=docs_path,
		env=env,
		check=True,
	)
	return docs_path / "_build_tag/html"


def main() -> None:
	"""Build all configured versions in isolation, then copy successful output."""
	main_commit = resolve_ref("refs/heads/main")
	with TemporaryDirectory(prefix="xtalmet-docs-") as temporary:
		workspace = Path(temporary)
		main_checkout = workspace / "main"
		checkout(main_checkout, main_commit)
		common_source = main_checkout / "docs/source"
		with (common_source / "versions.yaml").open() as yaml_file:
			versions = yaml.safe_load(yaml_file)
		builds = []
		for version, details in versions.items():
			if (
				not isinstance(version, str)
				or version in {"", ".", ".."}
				or Path(version).name != version
				or "\\" in version
			):
				raise ValueError(
					f"Version must be a single directory name: {version!r}"
				)
			builds.append((version, resolve_ref(details["tag"])))

		html = workspace / "html"
		shutil.copytree(build_doc("latest", main_checkout), html)
		for index, (version, commit) in enumerate(builds):
			tag_checkout = workspace / f"tag-{index}"
			checkout(tag_checkout, commit)
			tag_source = tag_checkout / "docs/source"
			for name in ("conf.py", "versions.yaml"):
				shutil.copy2(common_source / name, tag_source / name)
			tag_templates = tag_source / "_templates"
			if tag_templates.exists():
				shutil.rmtree(tag_templates)
			shutil.copytree(common_source / "_templates", tag_templates)
			shutil.copytree(build_doc(version, tag_checkout), html / version)
		output = PROJECT_ROOT / "docs/build/html"
		for destination in chain(
			(output, *output.parents),
			(output / path.relative_to(html) for path in html.rglob("*")),
		):
			if destination.is_symlink():
				raise ValueError(
					f"Refusing to write through output symlink: {destination}"
				)
		shutil.copytree(html, output, dirs_exist_ok=True)


if __name__ == "__main__":
	main()
