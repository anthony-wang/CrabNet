# Compositionally-Restricted Attention-Based Network (CrabNet)

The Compositionally-Restricted Attention-Based Network (`CrabNet`), inspired by natural language processing transformers, uses compositional information to predict material properties.

<img
src=https://user-images.githubusercontent.com/45469701/155030619-3a5f75e8-b28d-4801-a54c-58a800ee874c.png
width=150>

[![DOI](https://img.shields.io/badge/Paper:_npjCompuMat-10.1038%2Fs41524.021.00545.1-blue)](https://doi.org/10.1038/s41524-021-00545-1)

[![Open In Colab
(PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sparks-baird/CrabNet/blob/main/examples/crabnet_basic_colab.ipynb)
[![Read the Docs](https://img.shields.io/readthedocs/crabnet?label=Read%20the%20docs&logo=readthedocs)](https://crabnet.readthedocs.io/en/latest/)
[![GitHub Workflow
Status](https://img.shields.io/github/workflow/status/sparks-baird/mat_discover/Install%20with%20flit%20and%20test%20via%20Pytest?label=main)](https://github.com/sparks-baird/mat_discover/actions/workflows/flit-install-test.yml)

![PyPI](https://img.shields.io/pypi/v/crabnet) [![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Lines of code](https://img.shields.io/tokei/lines/github/sparks-baird/CrabNet)
![GitHub](https://img.shields.io/github/license/sgbaird/CrabNet)

[![Conda](https://img.shields.io/conda/v/sgbaird/crabnet)](https://anaconda.org/sgbaird/crabnet) [![Conda](https://img.shields.io/conda/pn/sgbaird/crabnet)](https://anaconda.org/sgbaird/crabnet) [![Conda](https://img.shields.io/conda/dn/sgbaird/crabnet?label=conda%7Cdownloads)](https://anaconda.org/sgbaird/crabnet) [![Anaconda-Server Badge](https://anaconda.org/sgbaird/crabnet/badges/latest_release_relative_date.svg)](https://anaconda.org/sgbaird/crabnet)

> :warning: This is a fork of the [original CrabNet repository](https://github.com/anthony-wang/CrabNet) :warning:

This is a refactored version of CrabNet, published to PyPI (`pip`) and Anaconda
(`conda`). In addition to using `.csv` files, it allows direct passing of Pandas
DataFrames as training and validation datasets, similar to
[automatminer](https://hackingmaterials.lbl.gov/automatminer/). It also exposes many of
the model parameters at the top-level via `CrabNet` and uses the `sklearn`-like "instantiate, fit, predict" workflow. An `extend_features` is
implemented which allows utilization of data other than the elemental compositions (e.g.
state variables such as temperature or applied load). These changes make CrabNet
portable, extensible, and more broadly applicable, and will be incorporated into the parent repository at a later
date. Please refer to the [CrabNet documentation](https://crabnet.readthedocs.io) for details on installation and usage. If you find CrabNet useful, please consider citing the [following publication](https://doi.org/10.1038/s41524-021-00545-1) in npj Computational Materials:

## Citing

```bibtex
@article{Wang2021crabnet,
 author = {Wang, Anthony Yu-Tung and Kauwe, Steven K. and Murdock, Ryan J. and Sparks, Taylor D.},
 year = {2021},
 title = {Compositionally restricted attention-based network for materials property predictions},
 pages = {77},
 volume = {7},
 number = {1},
 doi = {10.1038/s41524-021-00545-1},
 publisher = {{Nature Publishing Group}},
 shortjournal = {npj Comput. Mater.},
 journal = {npj Computational Materials}
}
```
