```{include} ../../README.md
:relative-images:
```

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
train_df["state_var0"] = np.random.rand(train_df.shape[0]) # dummy state variable
cb = CrabNet(
    mat_prop="hardness",
    train_df=train_df, # contains "formula", "target", and "state_var0" columns
    extend_features=["state_var0"],
    )
```

## Reproduce publication results

To reproduce the publication results, please see the README instructions for CrabNet versions v1.*.* or earlier. For example, the first release: https://github.com/sparks-baird/CrabNet/releases/tag/release-for-chemrxiv. **Trained weights are provided at:** <http://doi.org/10.5281/zenodo.4633866>.

As a reference, with a desktop computer with an Intel<sup>TM</sup> i9-9900K processor, 32GB of RAM, and two NVIDIA RTX 2080 Ti's, training our largest network (OQMD) takes roughly two hours.
