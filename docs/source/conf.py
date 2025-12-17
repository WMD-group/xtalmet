# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

## -- For autosummary -------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xtalmet"
copyright = "2025, Masahiro Negishi"
author = "Masahiro Negishi"
release = "1.0.0a2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.duration",
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",
	"sphinx_github_style",
	"sphinx.ext.githubpages",
]
autosummary_generate = True
autoclass_content = "both"
linkcode_url = "https://github.com/WMD-group/xtalmet"
autodoc_member_order = "bysource"


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False

import os

import yaml

# get the environment variable build_all_docs and pages_root
build_all_docs = os.environ.get("build_all_docs")
pages_root = os.environ.get("pages_root", "")

# if not there, we dont call this
if build_all_docs is not None:
	# we get the current language and version
	current_version = os.environ.get("current_version")

	# we set the html_context wit current language and version
	# and empty languages and versions for now
	html_context = {
		"current_version": current_version,
		"versions": [],
	}
	html_context["versions"].append(["latest", pages_root])

	# and loop over all other versions from our yaml file
	# to set versions and languages
	with open("versions.yaml") as yaml_file:
		docs = yaml.safe_load(yaml_file)

	for version, details in docs.items():
		html_context["versions"].append([version, pages_root + "/" + version])
