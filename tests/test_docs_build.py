"""Check isolated documentation builds without crystal models or network access."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "docs/build_docs.py"


def git(repo, *args):
	"""Run Git only in a disposable test repository."""
	return subprocess.check_output(
		["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
	).strip()


def commit(repo):
	"""Commit fixture changes without configuring the user's Git identity."""
	git(repo, "add", ".")
	git(
		repo,
		"-c",
		"user.name=Docs Test",
		"-c",
		"user.email=docs@example.test",
		"commit",
		"-qm",
		"fixture",
	)


def write(repo, name, content):
	"""Write a fixture file, creating its parent directories."""
	path = repo / name
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content)


@pytest.fixture
def docs_repo(tmp_path):
	"""Create distinct main/tag sources plus staged, unstaged and ignored data."""
	repo = tmp_path / "repo with spaces"
	repo.mkdir()
	git(repo, "init", "-qb", "main")
	write(
		repo, ".gitignore", "docs/build/\ndocs/source/api/\ndocs/_build_tag/\ndata/\n"
	)
	write(repo, "tracked.txt", "original\n")
	write(repo, "docs/build_docs.py", SCRIPT.read_text())
	config = (
		"import os\n"
		"from pathlib import Path\n"
		"import yaml\n"
		'project = "fixture"\n'
		'extensions = ["sphinx.ext.autodoc", "sphinx.ext.githubpages"]\n'
		'templates_path = ["_templates"]\n'
		'html_theme = "basic"\n'
		'html_title = "CONFIG_MARKER"\n'
		'html_context = {"current_version": os.environ.get("CURRENT_VERSION"),\n'
		'"versions": list(yaml.safe_load(Path("versions.yaml").read_text()))}\n'
	)
	for marker in ("TAG", "MAIN"):
		write(repo, "src/xtalmet/__init__.py", f'"""{marker}_PACKAGE."""\n')
		write(repo, "src/xtalmet/example.py", f'"""{marker}_MODULE."""\n')
		write(
			repo,
			"docs/source/index.rst",
			f"{marker}_TEXT\n=========\n\n.. toctree::\n\n   api/xtalmet\n",
		)
		write(
			repo,
			"docs/source/conf.py",
			config.replace("CONFIG_MARKER", marker + "_CONFIG"),
		)
		write(
			repo,
			"docs/source/versions.yaml",
			yaml.safe_dump({"0.1": {"tag": "v0.1"}} if marker == "MAIN" else {}),
		)
		write(
			repo,
			"docs/source/_templates/layout.html",
			'{% extends "!layout.html" %}\n{% block footer %}'
			+ marker
			+ "_TEMPLATE {{ current_version }} {{ versions|join(',') }}"
			+ "{{ super() }}{% endblock %}\n",
		)
		write(
			repo,
			"docs/source/_templates/apidoc/module.rst.jinja",
			marker + "_APIDOC\n\n.. automodule:: {{ qualname }}\n   :members:\n",
		)
		if marker == "TAG":
			write(
				repo,
				"docs/source/_templates/page.html",
				'{% extends "!page.html" %}{% block body %}'
				"TAG_ONLY_TEMPLATE{{ super() }}{% endblock %}\n",
			)
		else:
			(repo / "docs/source/_templates/page.html").unlink()
		commit(repo)
		if marker == "TAG":
			git(repo, "tag", "v0.1")
	git(repo, "checkout", "-qb", "feature")
	write(repo, "tracked.txt", "staged\n")
	git(repo, "add", "tracked.txt")
	write(repo, "tracked.txt", "unstaged\n")
	write(repo, "docs/source/conf.py", 'raise RuntimeError("dirty config")\n')
	write(repo, "local.txt", "untracked data\n")
	write(repo, "data/result.bin", "ignored result\n")
	write(repo, "docs/source/api/local.rst", "local API\n")
	write(repo, "docs/_build_tag/html/local.html", "local build\n")
	write(repo, "docs/build/html/index.html", "previous HTML\n")
	write(repo, "docs/build/html/keep.txt", "unrelated HTML directory file\n")
	return repo


def snapshot(repo, include_html=True):
	"""Capture refs, index changes and file contents, including ignored inputs."""
	files = {
		str(path.relative_to(repo)): path.read_bytes()
		for path in repo.rglob("*")
		if path.is_file()
		and ".git" not in path.relative_to(repo).parts
		and (include_html or not path.is_relative_to(repo / "docs/build/html"))
	}
	return (
		git(repo, "rev-parse", "HEAD"),
		git(repo, "for-each-ref"),
		git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
		git(repo, "diff", "--binary"),
		git(repo, "diff", "--cached", "--binary"),
		git(repo, "worktree", "list", "--porcelain"),
		files,
	)


@pytest.fixture
def builder(docs_repo, monkeypatch):
	"""Load the real builder against a disposable source repository."""
	spec = importlib.util.spec_from_file_location("docs_builder", SCRIPT)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	monkeypatch.setattr(module, "PROJECT_ROOT", docs_repo)
	return module


def test_import_has_no_side_effects(monkeypatch):
	"""Importing the module must not run commands or mutate process state."""
	before_env = dict(os.environ)
	before_cwd = Path.cwd()

	def unexpected_command(*args, **kwargs):
		pytest.fail("import ran an external command")

	monkeypatch.setattr(subprocess, "run", unexpected_command)
	spec = importlib.util.spec_from_file_location("docs_builder_import", SCRIPT)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	assert dict(os.environ) == before_env
	assert Path.cwd() == before_cwd


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("from_docs", [False, True])
def test_cli_preserves_source_and_builds_distinct_versions(
	docs_repo, tmp_path, detached, from_docs
):
	"""Real Sphinx builds use committed sources and leave dirty checkouts intact."""
	if detached:
		git(docs_repo, "checkout", "--detach", "-q")
	before = snapshot(docs_repo, include_html=False)
	scratch = tmp_path / "scratch"
	scratch.mkdir()
	cwd = docs_repo / "docs" if from_docs else docs_repo
	script = "build_docs.py" if from_docs else "docs/build_docs.py"
	result = subprocess.run(
		[sys.executable, script],
		cwd=cwd,
		env={**os.environ, "TMPDIR": str(scratch)},
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, result.stdout + result.stderr
	assert snapshot(docs_repo, include_html=False) == before
	output = docs_repo / "docs/build/html"
	for version, marker in (("", "MAIN"), ("0.1", "TAG")):
		html = (output / version / "index.html").read_text()
		api = (output / version / "api/xtalmet.html").read_text()
		assert marker + "_TEXT" in html
		assert marker + "_PACKAGE" in api
		assert "MAIN_CONFIG" in html
		assert "TAG_ONLY_TEMPLATE" not in html
		assert "MAIN_TEMPLATE " + (version or "latest") + " 0.1" in html
		module_api = (output / version / "api/xtalmet.example.html").read_text()
		assert "MAIN_APIDOC" in module_api
		assert marker + "_MODULE" in module_api
		assert (output / version / ".nojekyll").exists()
	assert (output / "keep.txt").read_text() == "unrelated HTML directory file\n"
	assert not list(scratch.iterdir())


@pytest.mark.parametrize("missing_ref", ["refs/heads/main", "refs/tags/v0.1"])
def test_missing_ref_fails_cli_without_changing_source(
	docs_repo, tmp_path, missing_ref
):
	"""Missing local references must fail without publishing or changing inputs."""
	git(docs_repo, "update-ref", "-d", missing_ref)
	before = snapshot(docs_repo)
	result = subprocess.run(
		[sys.executable, "docs/build_docs.py"],
		cwd=docs_repo,
		env={**os.environ, "TMPDIR": str(tmp_path)},
		capture_output=True,
		text=True,
	)
	assert result.returncode != 0
	assert "CalledProcessError" in result.stderr
	assert snapshot(docs_repo) == before
	assert not list(tmp_path.glob("xtalmet-docs-*"))


@pytest.mark.parametrize("failed_module", ["sphinx.ext.apidoc", "sphinx"])
def test_tag_failure_keeps_previous_html_and_cleans_clones(
	builder, docs_repo, monkeypatch, failed_module
):
	"""A failed tag build must not publish even a successfully built latest."""
	before = snapshot(docs_repo)
	before_env = dict(os.environ)
	before_cwd = Path.cwd()
	run = subprocess.run
	workspaces = []

	def fail_tag(args, **kwargs):
		if args[:2] == ["git", "clone"]:
			workspaces.append(Path(args[-1]).parent)
		if (
			args[:3] == [sys.executable, "-m", failed_module]
			and kwargs["env"]["CURRENT_VERSION"] == "0.1"
		):
			assert kwargs["check"] is True
			raise subprocess.CalledProcessError(23, args)
		return run(args, **kwargs)

	monkeypatch.setattr(subprocess, "run", fail_tag)
	with pytest.raises(subprocess.CalledProcessError) as error:
		builder.main()
	assert error.value.returncode == 23
	assert error.value.cmd[2] == failed_module
	assert snapshot(docs_repo) == before
	assert dict(os.environ) == before_env
	assert Path.cwd() == before_cwd
	assert workspaces and all(not path.exists() for path in workspaces)


@pytest.mark.parametrize(
	"version", ["../outside", "/outside", r"..\outside", ".", "", 1]
)
def test_invalid_output_version_fails_before_build(
	builder, docs_repo, monkeypatch, version
):
	"""Version labels cannot escape the generated output directory."""
	git(docs_repo, "checkout", "main", "--", "docs/source/conf.py")
	write(
		docs_repo,
		"docs/source/versions.yaml",
		yaml.safe_dump({version: {"tag": "v0.1"}}),
	)
	commit(docs_repo)
	git(docs_repo, "update-ref", "refs/heads/main", git(docs_repo, "rev-parse", "HEAD"))
	before = snapshot(docs_repo)

	def unexpected_build(*args):
		pytest.fail("invalid versions must fail before Sphinx runs")

	monkeypatch.setattr(builder, "build_doc", unexpected_build)
	with pytest.raises(ValueError, match="single directory name"):
		builder.main()
	assert snapshot(docs_repo) == before


@pytest.fixture
def fake_builds(builder, monkeypatch):
	"""Produce tiny successful builds for destination safety checks."""

	def build(version, checkout_path):
		write(checkout_path, "fake-html/index.html", version + " generated HTML\n")
		return checkout_path / "fake-html"

	monkeypatch.setattr(builder, "build_doc", build)


@pytest.mark.parametrize(
	"collision",
	[
		"docs",
		"docs/build",
		"docs/build/html",
		"docs/build/html/0.1",
		"docs/build/html/index.html",
	],
)
def test_output_symlink_cannot_overwrite_inputs(
	builder, fake_builds, docs_repo, tmp_path, collision
):
	"""Reject generated destination symlinks before writing any output."""
	write(docs_repo, "docs/build/html/0.1/index.html", "previous tag HTML\n")
	protected = tmp_path / "protected"
	protected.mkdir()
	destination = docs_repo / collision
	destination.rename(protected / "original")
	destination.symlink_to(
		protected / "original", target_is_directory=(protected / "original").is_dir()
	)
	before = snapshot(docs_repo)
	protected_before = {
		path.relative_to(protected): path.read_bytes()
		for path in protected.rglob("*")
		if path.is_file()
	}
	with pytest.raises(ValueError, match="output symlink"):
		builder.main()
	assert snapshot(docs_repo) == before
	assert destination.is_symlink()
	assert {
		path.relative_to(protected): path.read_bytes()
		for path in protected.rglob("*")
		if path.is_file()
	} == protected_before


def test_unrelated_output_symlink_is_retained(
	builder, fake_builds, docs_repo, tmp_path
):
	"""Do not touch unrelated symlinks when merging successful output."""
	protected = tmp_path / "protected.txt"
	protected.write_text("protected data\n")
	link = docs_repo / "docs/build/html/unrelated.txt"
	link.symlink_to(protected)
	before = snapshot(docs_repo, include_html=False)
	builder.main()
	assert snapshot(docs_repo, include_html=False) == before
	assert link.is_symlink()
	assert protected.read_text() == "protected data\n"
	assert (docs_repo / "docs/build/html/index.html").read_text() == (
		"latest generated HTML\n"
	)
