Installation
============

xtalmet can be installed via pip. It is recommended to use a virtual environment.

.. code-block:: console

   (.venv) $ pip install xtalmet

Compatibility
-------------

Python 3.12 and 3.13 are supported. The package requires Torch >=2.7.0;
CPU or CUDA support depends on the installed Torch build. Choose the build
appropriate for your hardware using the `PyTorch installation instructions
<https://pytorch.org/get-started/locally/>`_.

For development from a source checkout, reproduce the locked environment with:

.. code-block:: console

   $ uv sync --locked --group dev

The lockfile records one resolved development environment. Package dependency
ranges are declared in ``pyproject.toml``.

The following configurations were checked on Linux x86_64:

.. list-table:: Tested configurations
   :header-rows: 1

   * - Python
     - Torch
     - NumPy
     - SciPy
   * - 3.12.3
     - 2.7.0 / 2.8.0 (CPU)
     - 2.3.3
     - 1.16.2
   * - 3.12.3
     - 2.8.0+cu128 (CPU / CUDA 12.8)
     - 2.3.3
     - 1.16.2
   * - 3.13.16
     - 2.14.1 (CPU / CUDA 12.6)
     - 2.5.3
     - 1.18.1

These checks used SMACT 4.0.2 and pymatgen 2025.10.7, including all eight
distances and MACE-based VSUN evaluation with ``medium-mpa-0`` and ``mh-1``.
The tested versions describe verification results; the dependency ranges
remain those in ``pyproject.toml``.

The configurations above were checked at different implementation stages.
The final-source CPU suite on October 22, 2026 had 808 passes and one live-MP-API
skip. A separate CPU/CUDA comparison used Python 3.12.3, Torch 2.8.0+cu128,
10 generated and 10 reference structures, both MACE models and all eight
distances on an NVIDIA RTX 4000 Ada Generation. Hull
energies and VSUN scores agreed within ``rtol=atol=1e-10``.

A focused final-source CPU check with Python 3.13.16, Torch 2.14.1+cpu,
NumPy 2.5.3 and SciPy 1.18.1 passed 388 regression tests. The same small VSUN
comparison with both models and all eight distances matched the Python 3.12 CPU
results exactly. The full Python 3.13 suite and Torch 2.7.0 were not rerun on the
final source.

Materials Project entry decoding and phase-diagram regression tests also passed
with pymatgen 2026.9.24 and pymatgen-core 2026.10.16. The live MP loader handles
entry serialization from before and after the pymatgen-core split.

SMACT screening now uses SMACT >=4.0.2 defaults: ICSD24 oxidation states,
consensus 3, commonality ``low``, and mixed valence disabled. SMACT 3 used
commonality ``medium``, so validity and VSUN scores can change after upgrading.
