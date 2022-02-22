# Compositionally-Restricted Attention-Based Network (CrabNet)

The Compositionally-Restricted Attention-Based Network (`CrabNet`), inspired by natural language processing transformers, uses compositional information to predict material properties.

| <img src=https://user-images.githubusercontent.com/45469701/155030619-3a5f75e8-b28d-4801-a54c-58a800ee874c.png width=150> | [![Open In Colab (PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sparks-baird/CrabNet) [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/sparks-baird/mat_discover/Install%20with%20flit%20and%20test%20via%20Pytest?label=main)](https://github.com/sparks-baird/mat_discover/actions/workflows/flit-install-test.yml) <br /> ![PyPI](https://img.shields.io/pypi/v/crabnet) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Lines of code](https://img.shields.io/tokei/lines/github/sparks-baird/CrabNet) ![GitHub](https://img.shields.io/github/license/sgbaird/CrabNet) <br /> [![Conda](https://img.shields.io/conda/v/sgbaird/crabnet)](https://anaconda.org/sgbaird/crabnet) [![Conda](https://img.shields.io/conda/pn/sgbaird/crabnet)](https://anaconda.org/sgbaird/crabnet) [![Conda](https://img.shields.io/conda/dn/sgbaird/crabnet?label=conda%7Cdownloads)](https://anaconda.org/sgbaird/crabnet) [![Anaconda-Server Badge](https://anaconda.org/sgbaird/crabnet/badges/latest_release_relative_date.svg)](https://anaconda.org/sgbaird/crabnet) <br /> [![DOI](https://img.shields.io/badge/Paper:_npjCM-10.1038%2Fs41524.021.00545.1-blue)](https://doi.org/10.1038/s41524-021-00545-1) |
| --- | :-- |

> :warning: This is a fork of the [original CrabNet repository](https://github.com/anthony-wang/CrabNet) :warning:

This is a refactored version of CrabNet, published to PyPI (`pip`) and Anaconda (`conda`). In addition to using `.csv` files, it allows direct passing of Pandas DataFrames as training and validation datasets, similar to [automatminer](https://hackingmaterials.lbl.gov/automatminer/). It also exposes many of the model parameters at the top-level via `get_model`. An `extend_features` is also implemented which allows utilization of data other than the elemental compositions (e.g. state variables such as temperature or applied load). These changes make CrabNet portable and extensible, and may be incorporated into the parent repository at a later date. Basic instructions for this fork are given as follows, with the old documentation preserved towards the end.

## Installation

`conda install -c sgbaird crabnet`

or

[Install PyTorch](https://pytorch.org/get-started/locally/) (specific to your hardware, e.g. `pip install torch==1.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`), then

`pip install crabnet`

## Basic Usage

### Load Some Test Data

```python
from crabnet.model import get_data
from crabnet.data.materials_data import elasticity
train_df, val_df = get_data(elasticity, "train.csv")
```

### Instantiate CrabNet Model

```python
from crabnet.crabnet_ import CrabNet

cb = CrabNet(mat_prop="elasticity")
```

### Training

```python
cb.fit(train_df)
```

### Predictions

Predict on the training data:

```python
train_pred, train_sigma = cb.predict(train_df, return_uncertainty=True)
```

Predict on the validation data:

```python
val_pred, val_sigma = cb.predict(val_df)
```

### Extend Features

To include additional features that get added after the transformer architecture, but before a recurrent neural network, include the additional features in your DataFrames and pass the name(s) of these additional features (i.e. columns) as a list into `extend_features`.

```python
cb = CrabNet(
    mat_prop="hardness",
    train_df=train_df, # contains "formula", "target", and "state_var0" columns
    extend_features=["state_var0"],
    )
```

## How to cite

Please cite the following work if you want to use `CrabNet`:

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

## Reproduce publication results

To reproduce the publication results, please follow the below steps. Results will
slightly vary. It is a known phenomena that PyTorch model training may slightly vary on
different computers and hardware.

**Trained weights are provided at:** <http://doi.org/10.5281/zenodo.4633866>.

As a reference, with a desktop computer with an Intel<sup>TM</sup> i9-9900K processor, 32GB of RAM, and two NVIDIA RTX 2080 Ti's, training our largest network (OQMD) takes roughly two hours.

### Train CrabNet using files

1. To train crabnet you need `train.csv`, `val.csv`, and optionally a `test.csv` files.

   1. `train.csv` is used to find model weights.
  
   1. `val.csv` ensures the model does not overfit.
  
   1. `test.csv` will be run on the trained model for performance evaluation.
  
1. Place the csv files in the `data/materials_data` directory.

   1. The csv file must contain two columns, `formula` and `target`.
  
   1. `formula` must be a string containing valid element symbols, numbers, and parentheses.
  
   1. `target` is the target material property and should be provided as a number.
  
   1. Additional csv files can be saved here. In the case of inference with no known targets, you may fill the target columns with 0's.
  
1. Run `train_crabnet.py` to train CrabNet using default parameters.

   * If you desire to perform inference with additional csv files, you may add code to `train_crabnet.py` of the form
  
 ```python
 _, mae_added_data = save_results(data_dir, mat_prop, classification,
                                     'my_added_data.csv', verbose=False)
    ```

4. Note that your trained network will be associated with your given `mat_prop` folder.
If you want to predict with this model, you must use the same `mat_prop`.

### Plot results

1. Inference outputs using the provided saved weights are in the `predictions` folder.
1. Data are in the folder `publication_predictions`
1. Run `Paper_{FIG|TABLE}_{X}.py` to produce the tables and figures shown in the manuscript.

## IMPORTANT - if you want to reproduce the publication Figures 1 and 2

The PyTorch-builtin function for outting the multi-headed attention operation defaults to averaging the attention matrix across all heads.
Thus, in order to obtain the per-head attention information, we have to edit a bit of PyTorch's source code so that the individual attention matrices are returned.

To properly export the attention heads from the PyTorch `nn.MultiheadAttention` implementation within the transformer encoder layer, you will need to manually modify some of the source code of the PyTorch library.
This applies to PyTorch v1.6.0, v1.7.0, and v1.7.1 (potentially to other untested versions as well).

For this, open the file:
`C:\Users\{USERNAME}\Anaconda3\envs\{ENVIRONMENT}\Lib\site-packages\torch\nn\functional.py`
(where `USERNAME` is your Windows user name and `ENVIRONMENT` is your conda environment name (if you followed the steps above, then it should be `crabnet`))

At the end of the function defition of `multi_head_attention_forward` (line numbers may differ slightly):

```python
L4011 def multi_head_attention_forward(
# ...
# ... [some lines omitted]
# ...
L4291    if need_weights:
L4292        # average attention weights over heads
L4293        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
L4294        return attn_output, attn_output_weights.sum(dim=1) / num_heads
L4295    else:
L4296        return attn_output, None
```

Change the specific line

```python
return attn_output, attn_output_weights.sum(dim=1) / num_heads
```

to:

```python
return attn_output, attn_output_weights
```

This prevents the returning of the attention values as an average value over all heads, and instead returns each head's attention matrix individually.
For more information see:

* <https://github.com/pytorch/pytorch/issues/34537>
* <https://github.com/pytorch/pytorch/issues/32590>
* <https://discuss.pytorch.org/t/getting-nn-multiheadattention-attention-weights-for-each-head/72195/>
