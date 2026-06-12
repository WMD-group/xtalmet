"""Shared constants for journal revision notebooks."""

from __future__ import annotations

import math
from pathlib import Path


def _find_journal_dir() -> Path:
	"""Find the examples/journal directory from common notebook launch locations."""
	cwd = Path.cwd().resolve()
	candidates = [
		cwd,
		cwd.parent,
		cwd.parent.parent,
		cwd / "examples" / "journal",
	]
	for candidate in candidates:
		if (candidate / "results" / "mp20").exists():
			return candidate
	raise FileNotFoundError("Could not locate examples/journal/results/mp20")


JOURNAL_DIR = _find_journal_dir()
REVISION_DIR = JOURNAL_DIR / "revision"
RESULTS_DIR = JOURNAL_DIR / "results" / "mp20"
FIGURES_DIR = REVISION_DIR / "figures"
PREPROCESS_DIR = REVISION_DIR / "preprocess"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
	"cdvae",
	"diffcsp",
	"diffcsppp",
	"mattergen",
	"chemeleon",
	"adit",
	"chemeleon2",
]
MODELS_TEST = MODELS + ["test"]

BINARY_DISTANCES = ["smat", "comp", "wyckoff"]
CONTINUOUS_DISTANCES = ["elmd", "amd", "elmd+amd"]
NORMALIZATION_DISTANCES = ["amd", "elmd"]
DISTANCES = NORMALIZATION_DISTANCES

METRICS = {"uni": "Uniqueness", "nov": "Novelty"}
SCORE_COLUMNS = {"uni": "Uniqueness score", "nov": "Novelty score"}

SQRT_10 = math.sqrt(10)
NORMALIZATIONS = [
	{
		"key": "frac_c1",
		"label": r"$\frac{d^\prime}{1+d^\prime}$",
		"family": "frac",
		"param": 1.0,
	},
	{
		"key": "frac_csqrt10",
		"label": r"$\frac{d^\prime}{\sqrt{10}+d^\prime}$",
		"family": "frac",
		"param": SQRT_10,
	},
	{
		"key": "frac_c10",
		"label": r"$\frac{d^\prime}{10+d^\prime}$",
		"family": "frac",
		"param": 10.0,
	},
	{"key": "exp_l1", "label": r"$1-\exp(-d^\prime)$", "family": "exp", "param": 1.0},
	{
		"key": "exp_lsqrt10",
		"label": r"$1-\exp(-\frac{d^\prime}{\sqrt{10}})$",
		"family": "exp",
		"param": SQRT_10,
	},
	{
		"key": "exp_l10",
		"label": r"$1-\exp(-\frac{d^\prime}{10})$",
		"family": "exp",
		"param": 10.0,
	},
]
NORMALIZATION_KEYS = [norm["key"] for norm in NORMALIZATIONS]
NORMALIZATION_LABELS = {norm["key"]: norm["label"] for norm in NORMALIZATIONS}

ELMD_AMD_DEFAULT_WEIGHT = float.fromhex("0x1.8d7d565a99f87p-1")
ELMD_AMD_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
STABILITY_INTERCEPT = 0.4289

MODEL_TEST_LABELS = {
	"cdvae": "CDVAE",
	"diffcsp": "DiffCSP",
	"diffcsppp": "DiffCSP++",
	"mattergen": "MatterGen",
	"chemeleon": "Chemeleon-DNG",
	"adit": "ADiT",
	"chemeleon2": "Chemeleon2",
	"test": "MP20 Test Set",
}
U_LABELS = {
	"amd": r"$\overline{\mathrm{U}}_\mathrm{am}$",
	"elmd": r"$\overline{\mathrm{U}}_\mathrm{elm}$",
	"elmd+amd": r"$\overline{\mathrm{U}}_\mathrm{elm{+}am}$",
}
N_LABELS = {
	"amd": r"$\overline{\mathrm{N}}_\mathrm{am}$",
	"elmd": r"$\overline{\mathrm{N}}_\mathrm{elm}$",
	"elmd+amd": r"$\overline{\mathrm{N}}_\mathrm{elm{+}am}$",
}
SUN_LABELS = {
	"amd": r"$\overline{\mathrm{SUN}}_\mathrm{am}$",
	"elmd": r"$\overline{\mathrm{SUN}}_\mathrm{elm}$",
	"elmd+amd": r"$\overline{\mathrm{SUN}}_\mathrm{elm{+}am}$",
}
METRIC_LABEL_DICTS = {"uni": U_LABELS, "nov": N_LABELS, "sun": SUN_LABELS}

NORMALIZATION_PLOT_SETTINGS = [
	("uni", "amd"),
	("uni", "elmd"),
	("nov", "amd"),
	("nov", "elmd"),
]
ELMD_AMD_WEIGHT_METRICS = ["uni", "nov", "sun"]

SAMPLE_SCORES_PATH = PREPROCESS_DIR / "normalized_sample_scores.csv"
MEAN_SCORES_PATH = PREPROCESS_DIR / "normalized_model_means.csv"
SPEARMAN_PATH = PREPROCESS_DIR / "normalization_spearman.csv"
ELMD_AMD_WEIGHT_SAMPLE_SCORES_PATH = (
	PREPROCESS_DIR / "elmd_amd_weight_sample_scores.csv"
)
ELMD_AMD_WEIGHT_MEAN_SCORES_PATH = PREPROCESS_DIR / "elmd_amd_weight_model_means.csv"

GRAY = "#9FA0A0"
BLACK = "#000000"
WHITE = "#FFFFFF"
PALETTE = [
	"#316745",
	"#F39800",
	"#2CA9E1",
	"#c53d43",
	"#B8D200",
	"#19448E",
	"#884898",
	GRAY,
]

FIG_WIDTH = 6.0248
FONT_L = 10
FONT_M = 8
FONT_S = 6
PAD_M = 7
PAD_L = 10
PAD_S = 4
PAD_SS = 2
OVERWRITE = False
CHUNK_ROWS = 256
