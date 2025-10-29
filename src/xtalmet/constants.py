"""Type definitions used in xtalmet."""

import numpy as np

TYPE_EMB_COMP = tuple[
	tuple[str, int]
]  #: Type of embeddings for d_comp. A tuple containing elements and their counts (divided by gcd).

TYPE_EMB_WYCKOFF = (
	tuple[int, tuple[str]] | Exception
)  #: Type of embeddings for d_wyckoff. A tuple containing the space group number and a tuple of Wyckoff letters, or an Exception from SpacegroupAnalyzer.

TYPE_EMB_MAGPIE = list[
	float
]  #: Type of embeddings for d_magpie. A list of floats (Magpie feature vector).

TYPE_EMB_PDD = (
	np.ndarray[np.float32 | np.float64] | Exception
)  #: Type of embeddings for d_pdd. A numpy array of floats (pair distance distribution) or an Exception from periodicset_from_pymatgen_structure.

TYPE_EMB_AMD = (
	np.ndarray[np.float32 | np.float64] | Exception
)  #: Type of embeddings for d_amd. A numpy array of floats (average minimum distance) or an Exception from periodicset_from_pymatgen_structure.

TYPE_EMB_ALL = (
	TYPE_EMB_COMP | TYPE_EMB_WYCKOFF | TYPE_EMB_MAGPIE | TYPE_EMB_PDD | TYPE_EMB_AMD
)  #: Union type of all embeddings.

DIST_WO_EMB = ["smat"]  #: Distance metrics that do not use embeddings.

BINARY_DISTANCES = ["smat", "comp", "wyckoff"]  #: Binary distance metrics.
CONTINUOUS_DISTANCES = ["magpie", "pdd", "amd"]  #: Continuous distance metrics.
SUPPORTED_DISTANCES = (
	BINARY_DISTANCES + CONTINUOUS_DISTANCES
)  #: Supported distance metrics.

SUPPORTED_SCREENS = [None, "smact", "ehull"]  #: Supported screening methods.

HF_VERSION = "v1.0.0"  #: Version of Hugging Face repository to use
