"""Type definitions used in xtalmet."""

import numpy as np

TYPE_EMB_COMP = tuple[
	tuple[str, int]
]  #: Type of embeddings for d_comp. A tuple containing elements and their counts (divided by gcd).

TYPE_EMB_WYCKOFF = tuple[
	int, tuple[str]
]  #: Type of embeddings for d_wyckoff. A tuple containing the space group number and a tuple of Wyckoff letters.

TYPE_EMB_MAGPIE = list[
	float
]  #: Type of embeddings for d_magpie. A list of floats (Magpie feature vector).

TYPE_EMB_PDD = np.ndarray[
	np.float32 | np.float64
]  #: Type of embeddings for d_pdd. A numpy array of floats (pair distance distribution) .

TYPE_EMB_AMD = np.ndarray[
	np.float32 | np.float64
]  #: Type of embeddings for d_amd. A numpy array of floats (average minimum distance).

TYPE_EMB_ELMD = str  #: Type of embeddings for d_elmd. A string (compositional formula).

TYPE_EMB_ALL = (
	TYPE_EMB_COMP
	| TYPE_EMB_WYCKOFF
	| TYPE_EMB_MAGPIE
	| TYPE_EMB_PDD
	| TYPE_EMB_AMD
	| TYPE_EMB_ELMD
)  #: Union type of all embeddings.

DIST_WO_EMB = ["smat"]  #: Distance metrics that do not use embeddings.

BINARY_DISTANCES = ["smat", "comp", "wyckoff"]  #: Binary distance metrics.
CONTINUOUS_DISTANCES = ["magpie", "pdd", "amd", "elmd"]  #: Continuous distance metrics.
CONTINUOUS_UNNORMALIZED_DISTANCES = [
	"magpie",
	"pdd",
	"amd",
	"elmd",
]  #: Continuous distance metrics that are not normalized to [0, 1].
SUPPORTED_DISTANCES = (
	BINARY_DISTANCES + CONTINUOUS_DISTANCES
)  #: Supported distance metrics.

SUPPORTED_VALIDITY = ["smact", "structure"]  #: Supported validity screening methods.

HF_VERSION = "v1.0.0"  #: Version of Hugging Face repository to use.
