# Compositionally-Restricted Attention-Based Network (CrabNet)

This software package implements the Compositionally-Restricted Attention-Based Network (`CrabNet`) that takes only composition information to predict material properties.



## Table of Contents
* How to cite
* Installation
* Reproduce publication results
* Train or predict materials properties using CrabNet or DenseNet



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


## Installation
This code uses PyTorch for creating the neural network models.
For fast model training and inference, it is suggested you use an NVIDIA GPU
with the most recent drivers.

Windows users should be able to install all required Python packages
via Anaconda by following the steps below.

Linux users will additionally need to manually install CUDA and cuDNN.


### Clone or download this GitHub repository
Do one of the following:

* [Clone this repository](https://github.com/anthony-wang/CrabNet.git)
    to a directory of your choice on your computer.
* [Download an archive of this repository](https://github.com/anthony-wang/CrabNet/archive/master.zip)
    and extract it to a directory of your choice on your computer.


### Install dependencies via Anaconda:
1. Download and install [Anaconda](https://conda.io/docs/index.html).
1. Navigate to the project directory (from above).
1. Open Anaconda prompt in this directory.
1. Run the following command from Anaconda prompt to automatically create
    an environment from the `conda-env.yml` file:
    - `conda env create --file conda-env.yml`
	- `conda env create --file conda-env-cpuonly.yml` if you only have a CPU and no GPU in your system
1. Run the following command from Anaconda prompt to activate the environment:
    - `conda activate crabnet`

For more information about creating, managing, and working with Conda environments, please consult the [relevant help page](https://conda.io/docs/user-guide/tasks/manage-environments.html).


### Install dependencies via `pip`:
Open `conda-env.yml` and `pip install` all of the packages listed there.
We recommend that you create a separate Python environment for this project.



## Reproduce publication results
To reproduce the publication results, please follow the below steps.
Results will slightly vary.
It is a known phenomena that PyTorch model training may slightly vary on different computers and hardware.

**Trained weights are provided at:** http://doi.org/10.5281/zenodo.4633866.

As a reference, with a desktop computer with an Intel<sup>TM</sup> i9-9900K processor, 32GB of RAM, and two NVIDIA RTX 2080 Ti's, training our largest network (OQMD) takes roughly two hours.


### Train CrabNet
1. To train crabnet you need `train.csv`, `val.csv`, and optionally a `test.csv` files. 
	- `train.csv` is used to find model weights.
	- `val.csv` ensures the model does not overfit.
	- `test.csv` will be run on the trained model for performance evaluation.
1. Place the csv files in the `data/materials_data` directory.
	- The csv file must contain two columns, `formula` and `target`.
	- `formula` must be a string containing valid element symbols, numbers, and parentheses.
	- `target` is the target material property and should be provided as a number.
	- Additional csv files can be saved here. In the case of inference with no known targets, you may fill the target columns with 0's.
1. Run `train_crabnet.py` to train CrabNet using default parameters. 
	- If you desire to perform inference with additional csv files, you may add code to `train_crabnet.py` of the form 
	```python
	_, mae_added_data = save_results(data_dir, mat_prop, classification,
                                     'my_added_data.csv', verbose=False)
    ```
1. Note that your trained network will be associated with your given `mat_prop` folder.
If you want to predict with this model, you must use the same `mat_prop`.


### Plot results
1. Inference outputs using the provided saved weights are in the `predictions` folder.
1. Data are in the folder `publication_predictions`
1. Run `Paper_{FIG|TABLE}_{X}.py` to produce the tables and figures shown in the manuscript.


## IMPORTANT - if you want to reproduce the publication Figures 1 and 2:
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
- https://github.com/pytorch/pytorch/issues/34537
- https://github.com/pytorch/pytorch/issues/32590
- https://discuss.pytorch.org/t/getting-nn-multiheadattention-attention-weights-for-each-head/72195/
