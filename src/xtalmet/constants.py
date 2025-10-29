"""Type definitions used in xtalmet."""

import numpy as np

TYPE_EMB_COMP = tuple[tuple[str, int]]
TYPE_EMB_WYCKOFF = tuple[int, tuple[str]] | Exception
TYPE_EMB_MAGPIE = list[float]
TYPE_EMB_PDD = np.ndarray[np.float32 | np.float64] | Exception
TYPE_EMB_AMD = np.ndarray[np.float32 | np.float64] | Exception
TYPE_EMB_ALL = (
	TYPE_EMB_COMP | TYPE_EMB_WYCKOFF | TYPE_EMB_MAGPIE | TYPE_EMB_PDD | TYPE_EMB_AMD
)

DIST_WO_EMB = ["smat"]

BINARY_DISTANCES = ["smat", "comp", "wyckoff"]
CONTINUOUS_DISTANCES = ["magpie", "pdd", "amd"]
SUPPORTED_DISTANCES = BINARY_DISTANCES + CONTINUOUS_DISTANCES

SUPPORTED_SCREENS = [None, "smact", "ehull"]

HF_VERSION = "v1.0.0"
