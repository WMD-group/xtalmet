Tutorial
========

A guide to basic usage of xtalmet is shown in `README.md <https://github.com/WMD-group/xtalmet?tab=readme-ov-file#usage>`_.
We also provide a more detailed tutorial notebook `here <https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb>`_.

Distances and evaluation
------------------------

Distance matrices remain dense float64 arrays. Failed embeddings produce NaN
rows and columns, including the diagonal; nonfinite pair distances are also NaN.
Failed generated embeddings receive zero uniqueness and novelty scores, while
invalid references are excluded. Novelty requires at least one valid reference;
continuous uniqueness requires at least two generated structures. See
:func:`~xtalmet.distance.distance_matrix` and
:meth:`~xtalmet.evaluator.Evaluator.evaluate` for the calculation and failure rules.

Set ``multiprocessing`` and ``n_processes`` explicitly for large inputs.
Constructor settings control reference preparation; ``evaluate()`` settings
control generated embeddings and matrices. PDD's ``args_dist["n_jobs"]`` overrides
the matrix worker count. A 10,000 by 30,000 matrix alone uses about 2.24 GiB;
component matrices, temporary arrays and workers require additional memory.

References and caches
---------------------

Use :meth:`~xtalmet.evaluator.Evaluator.save_reference_embeddings` to save custom
references with their distance and embedding settings. Legacy plain embedding
files must be regenerated from structures. Non-default MP20 embedding settings
issue a warning and recompute references from the pinned training structures
during initialization.

Passing ``dir_intermediate`` to ``evaluate()`` stores validated records under
``cache/format-1/<identity>/``. Reuse requires matching ordered inputs, relevant
settings, references, model/phase data and software. Old unversioned files are
preserved and ignored; invalid records are recomputed. The notebook demonstrates
reference-file round trips and describes cache reuse in more detail.
