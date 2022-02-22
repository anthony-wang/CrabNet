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
cb = CrabNet(
    mat_prop="hardness",
    train_df=train_df, # contains "formula", "target", and "state_var0" columns
    extend_features=["state_var0"],
    )
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
