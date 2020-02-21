# Compositionally-Restricted Attention-Based Network (CrabNet)

This software package implements the Compositionally-Restricted Attention-Based Network (`CrabNet`) that takes only composition information to predict material properties.



## Table of Contents
* How to cite
* Installation
* Reproduce publication results
* Train or predict materials properties using CrabNet or DenseNet



## How to cite
Please cite the following work if you use `CrabNet`:

Wang, Anthony Yu-Tung; Kauwe, Steven K.; Murdock, Ryan J.; Sparks, Taylor D. (2020): [Compositionally-Restricted Attention-Based Network for Materials Property Prediction](https://doi.org/10.26434/chemrxiv.11869026.v1). ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.11869026

BibTeX format:
```bibtex
@article{Wang2020CrabNet,
  author = "Wang, Anthony Yu-Tung and Kauwe, Steven K. and Murdock, Ryan J. and Sparks, Taylor D.",
  title = "{Compositionally-Restricted Attention-Based Network for Materials Property Prediction}",
  year = "2020",
  month = "2",
  url = "https://chemrxiv.org/articles/Compositionally-Restricted_Attention-Based_Network_for_Materials_Property_Prediction/11869026",
  doi = "10.26434/chemrxiv.11869026"
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
1. Run the following command from Anaconda prompt to activate the environment:
    - `conda activate crabnet`

For more information about creating, managing, and working with Conda environments, please consult the [relevant help page](https://conda.io/docs/user-guide/tasks/manage-environments.html).


### Install dependencies via `pip`:
Open `conda-env.yml` and `pip install` all of the packages listed there.
We recommend that you create a separate Python environment for this project.



## Reproduce publication results
To reproduce the publication results, please follow the below steps.

Note, this work evaluates model performance over seven different material properties for two different neural network architectures and seven different classical machine learning models (each with extensive gridsearch parameters).
Therefore, reproducing each section of this work may take on the scale of hours on a modern desktop computer.

As a reference, with a desktop computer with an Intel<sup>TM</sup> i9-9900K processor and an NVIDIA RTX 2080 Ti, training the neural networks with one random seed takes around 2 hours, and gridsearch over the classical machine learning models takes around 6.5 hours.

Training a single CrabNet model is significantly faster.

### Train neural networks (CrabNet and DenseNet)
1. Run `python train_ann.py` to train both CrabNet and DenseNet using the default parameters. 
	- This script uses a default seed of 9, `RNG_SEEDS = [9]`. 
	- In total, 10 seeds were tested, `RNG_SEEDS = [7, 9, 2, 15, 4, 0, 10, 1337, 11, 13]`.
1. Run `python copy_weights.py` to copy the saved weights to the `data/user_properties/trained_weights` directory

### Train classical ML models
1. Run `python train_classics.py` to conduct a full GridSearchCV over all classical model combinations using a large number of parameter combinations.
	- Figures showing the results of the GridSearchCV are saved in the `figures/GridSearchCV` directory.
1. Run `python retrain_classics.py` to retrain the best-performing classical model combinations on the combined training and validation dataset and evaluate their performance on the test dataset.
	- Figures showing the test performance of the best model combinations are saved in the `figures/Classics` directory.

### Collect metrics and plot results
1. Run `python publication_results_plots.py` to produce the results tables and publication plots.
	- Neural network (CrabNet and DenseNet) and classical model test metrics are stored in their respective run directories.
	- The figures are saved in the `data/learning_curves` and `data/pred_vs_act` directories.



## Train or predict materials properties using CrabNet or DenseNet
Consult the documentation in the `use_crabnet.py` and `use_densenet.py` files for additional details.



### Training CrabNet or DenseNet
1. Place the csv file ([example](data/material_properties/ael_bulk_modulus_vrh/train.csv)) containing compositions you want to train on inside the `data/user_properties` directory.
	- The csv file must contain two columns, `cif_id` and `target`. 
	- `cif_id` can simply be the chemical formula of a compound or can be in the format `formula_ICSD_NNNNN` where NNNNN is the CIF ID corresponding to this compound. (the CIF ID part is not used in this work).
	- `target` is the target material property.
1. Edit the lines inside `use_crabnet.py` or `use_densenet.py` that say `train_crabnet` or `train_densenet`, respectively, to reflect your custom model name, and the name of the csv file to train on.
1. Start the training by running `python use_crabnet.py` or `python use_densenet.py` (see below).

```python
train_crabnet(model_name='your_model_name',
              csv_train='example_bulk_train.csv')
```

Note that your trained network will be associated with your given `model_name='your_model_name'` from above.
If you want to predict with this model, you must use this same `model_name`.


### Predicting with CrabNet or DenseNet
1. Place the csv file ([example](data/material_properties/ael_bulk_modulus_vrh/train.csv)) containing compositions you want to predict on inside the `data/user_properties` directory.
	- The csv file must contain two columns, `cif_id` and `target`.
	- The `target` values can be filled with any values in the case of prediction.
1. Edit the lines inside `use_crabnet.py` or `use_densenet.py` to reflect your custom model name, and the name of the csv file to predict on (see below).
1. Start the prediction by running `python use_crabnet.py` or `python use_densenet.py`.


```python
input_values, predicted_values = predict_crabnet(model_name='your_model_name',
                                                 csv_pred='example_bulk_pred.csv')
```
