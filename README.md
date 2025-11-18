# xtalmet
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Formatter: Ruff](https://img.shields.io/badge/formatter-Ruff-D7FF64.svg?logo=ruff)](https://docs.astral.sh/ruff/)
[![Linter: Ruff](https://img.shields.io/badge/linter-Ruff-D7FF64.svg?logo=ruff)](https://docs.astral.sh/ruff/)
[![Packaging: uv](https://img.shields.io/badge/packaging-uv-DE5FE9.svg?logo=uv)](https://docs.astral.sh/uv/)
[![Docs: Sphinx](https://img.shields.io/badge/docs-Sphinx-000000.svg?logo=sphinx)](https://www.sphinx-doc.org/en/master/index.html)
[![GitHub issues](https://img.shields.io/github/issues-raw/WMD-Group/xtalmet)](https://github.com/WMD-group/xtalmet/issues)
[![CI Status](https://github.com/WMD-group/xtalmet/actions/workflows/sphinx.yml/badge.svg)](https://github.com/WMD-group/xtalmet/actions/workflows/sphinx.yml)

The **xtalmet** package offers a variety of distance functions for comparing crystal structures. 
These include both binary and continuous as well as compositional and structural functions. 
It also allows you to evaluate a set of crystals based on Validity, Stability, Uniqueness, and Novelty metrics, or any combination of them. 
The uniqueness and novelty evaluations depend on the selected distance function. 

- Documentation: https://wmd-group.github.io/xtalmet/
- Examples: https://github.com/WMD-group/xtalmet/examples

> **Note**: This package is under active development. Please ensure that the version of the package you are using matches the version of the documentation you are referring to.

## Motivation
A central challenge in materials science is the efficient discovery of functional crystals from the vast chemical space. 
Recently, the inverse design of crystals using advanced machine learning generative models has emerged as a promising approach due to their ability to rapidly generate numerous candidates. 
However, while these models have become increasingly sophisticated, their evaluation metrics have remained largely unchanged since the seminal work of Xie et al [1] and Zeni et al [2]. 
To effectively guide model development, these evaluation metrics must also continue to improve. 
We aim to refine two primary metrics, uniqueness and novelty, by revising the underlying distance function used to compare crystal structures.
Additionally, we introduce a method for continuously measuring the stability of crystals.

[1] Tian Xie et al. Crystal diffusion variational autoencoder for periodic material generation. International Conference on Learning Representations 2022.

[2] Claudio Zeni et al. A generative model for inorganic materials design. Nature 2025.

## Installation
The latest stable version can be installed via pip:
```bash
pip install xtalmet
```

## Usage
Two primary features of xtalmet are the calculation of distances between crystals and the VSUN evaluation.
For the former usage, suppose you have two crystals `xtal_1` and `xtal_2` (`pymatgen.core.Structure`) whose distance you want to measure.
You can do so with one line of code:
```python
from xtalmet.distance import distance
d = distance("amd", xtal_1, xtal_2)
```
Here, "amd" is a type of continuous distance based on structural fingerprints.
For a complete list of available distances, please refer to our [tutorial notebook](https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb).

For the VSUN evaluation, imagine that you want to assess a set of crystals `gen_xtals` (`list[pymatgen.core.Structure]`) generated from a model trained on the MP20 dataset.
This can be done with just a few lines of code:
```python
from xtalmet.evaluator import Evaluator
evaluator = Evaluator(validity=["smact", "structure"], stability="continuous", uniqueness=True, novelty=True, distance="elmd", ref_xtals="mp20")
vsun, _, _ = evaluator.evaluate(xtals=gen_xtals)
```
A more detailed tutorial notebook is provided [here](https://github.com/WMD-group/xtalmet/blob/main/examples/tutorial.ipynb).

## Acknowledgements
Although we have argued that the progress on evaluation metrics is slower than that on generative models, we are not the sole contributors to this field. 
Our work builds upon several influential studies. 
We particularly acknowledge the following contributions:

- Zeni et al. [2]: For their implementation of the S.U.N. (Stability / Uniqueness / Novelty) metric.

- Baird et al. [3]: For their code for uniqueness and novelty, and their framework for creating a Python benchmarking package.

- Widdowson et al. [4, 5]: For their work on a continuous structural distance between crystals.

- Hargreaves et al.'s [6]: For their work on an optimal transport-based methods for inorganic compositions.

Other studies, while not directly related, were also inspiring in shaping our approach to measuring distances between crystals.  
These include the work of Onwuli et al. [7] on distances between elements.

[3] Baird et al. matbench-genmetrics: A Python library for benchmarking crystal structure generative models using time-based splits of Materials Project structures. Journal of Open Source Software 2024.

[4] Widdowson et al. Resolving the data ambiguity for periodic crystals. Advances in Neural Information Processing Systems 2022.

[5] Widdson et al. Average Minimum Distances of periodic point sets - foundational invariants for mapping periodic crystals. MATCH Communications in Mathematical and in Computer Chemistry 2022.

[6] Hargreaves et al. The earth mover’s distance as a metric for the space of inorganic compositions. Chemistry of Materials 2020.

[7] A. Onwuli et al. Element similarity in high-dimensional materials representations. Digital Discovery 2023.

## Citation
If you find xtalmet useful in your research, please consider citing:
```bibtex
@inproceedings{negishi2025continuous,
      title={Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals},
      author={Masahiro Negishi and Hyunsoo Park and Kinga O. Mastej and Aron Walsh},
      booktitle={AI for Accelerated Materials Design - NeurIPS 2025},
      year={2025},
      url={https://openreview.net/forum?id=PiKMmLHbEH}
}
```