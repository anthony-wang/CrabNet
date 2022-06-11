```{include} ../../README.md
:relative-images:
```

## Installation

`conda install -c sgbaird crabnet`

or

[Install PyTorch](https://pytorch.org/get-started/locally/) (specific to your hardware, e.g. `pip install torch==1.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`), then

`pip install crabnet`

## Basic Usage

### Data
You can load data from a file, hard-code it, use a predefined dataset, or download directly from Materials Project. Here, we hard-code it. See the Data section below for instructions for other methods.

#### Hard-coded
For a quick hard-coded example, you could use:
```python
train_df = pd.DataFrame(dict(formula=["Tc1V1", "Cu1Dy1", "Cd3N2"], target=[248.539, 66.8444, 91.5034]))
val_df = pd.DataFrame(dict(formula=["Al2O3", "SiO2"]))
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

## Data

#### From File
If you're using your own dataset, you will need to supply a Pandas DataFrame that
contains `formula` (string) and `target` (numeric) columns. If you have a `train.csv` file
(located in current working directory) with these two columns, this can be converted to
a DataFrame via:

```python
import pandas as pd
train_df = pd.read_csv("train.csv")
```

which might look something like the following:

formula | target
---|---
Tc1V1 | 248.539
Cu1Dy1 | 66.8444
Cd3N2 | 91.5034

For validation data without known property values to be used with `predict`, dummy
values (all zeros) are assigned internally. In this case, you can read in a CSV file
that contains only the `formula` (string) column:

```python
val_df = pd.read_csv("val.csv")
```

| formula |
| --- |
| Al2O3 |
| SiO2 |

#### CrabNet Datasets (including Matbench)
NOTE: you can load any of the datasets within `CrabNet/data/`, which includes `matbench` data, other datasets from the CrabNet paper, and a recent (as of Oct 2021) snapshot of `K_VRH` bulk modulus data from Materials Project. For example, to load the bulk modulus snapshot:

```python
from crabnet.model import get_data
from crabnet.data.materials_data import elasticity
train_df, val_df = get_data(elasticity, "train.csv") # note that `val.csv` within `elasticity` is every other Materials Project compound (i.e. "target" column filled with zeros)
```

The built-in data directories are as follows:
>
> ```python
> {'benchmark_data',
>  'benchmark_data.CritExam__Ed',
>  'benchmark_data.CritExam__Ef',
>  'benchmark_data.OQMD_Bandgap',
>  'benchmark_data.OQMD_Energy_per_atom',
>  'benchmark_data.OQMD_Formation_Enthalpy',
>  'benchmark_data.OQMD_Volume_per_atom',
>  'benchmark_data.aflow__Egap',
>  'benchmark_data.aflow__ael_bulk_modulus_vrh',
>  'benchmark_data.aflow__ael_debye_temperature',
>  'benchmark_data.aflow__ael_shear_modulus_vrh',
>  'benchmark_data.aflow__agl_thermal_conductivity_300K',
>  'benchmark_data.aflow__agl_thermal_expansion_300K',
>  'benchmark_data.aflow__energy_atom',
>  'benchmark_data.mp_bulk_modulus',
>  'benchmark_data.mp_e_hull',
>  'benchmark_data.mp_elastic_anisotropy',
>  'benchmark_data.mp_mu_b',
>  'benchmark_data.mp_shear_modulus',
>  'element_properties',
>  'matbench',
>  'materials_data',
>  'materials_data.elasticity',
>  'materials_data.example_materials_property'}
> ```

To see what `.csv` files are available (e.g. `train.csv`), you will probably need to navigate to [CrabNet/data/](https://github.com/sgbaird/CrabNet/tree/master/crabnet/data) and explore. For example, to use a snapshot of the Materials Project `e_above_hull` dataset ([`mp_e_hull`](https://github.com/sgbaird/CrabNet/tree/master/crabnet/data/benchmark_data/mp_e_hull)):
```python
from crabnet.data.benchmark_data import mp_e_hull
train_df = disc.data(mp_e_hull, "train.csv", split=False)
val_df = disc.data(mp_e_hull, "val.csv", split=False)
test_df = disc.data(mp_ehull, "test.csv", split=False)
```

#### Directly via Materials Project
Finally, to download data from Materials Project directly, see [generate_elasticity_data.py](https://github.com/sparks-baird/mat_discover/blob/main/mat_discover/utils/generate_elasticity_data.py).

## Reproduce publication results

To reproduce the publication results, please see the README instructions for CrabNet versions v1.*.* or earlier. For example, the first release: https://github.com/sparks-baird/CrabNet/releases/tag/release-for-chemrxiv. **Trained weights are provided at:** <http://doi.org/10.5281/zenodo.4633866>.

As a reference, with a desktop computer with an Intel<sup>TM</sup> i9-9900K processor, 32GB of RAM, and two NVIDIA RTX 2080 Ti's, training our largest network (OQMD) takes roughly two hours.
