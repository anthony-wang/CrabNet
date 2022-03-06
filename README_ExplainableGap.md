# Model interpretability with `CrabNet`

Use these steps to explore model interpretability methods with `CrabNet` and to reproduce the results presented in the publication:

A. Y.-T. Wang, M. S. Mahmoud, M. Czasny, A. Gurlo, CrabNet for Explainable Deep Learning in Materials Science: Bridging the Gap Between Academia and Industry, *Integr. Mater. Manuf. Innov.*, **2022**, *11 (1)*: 41-56. DOI: [10.1007/s40192-021-00247-y](https://doi.org/10.1038/s41524-021-00545-1).



## Table of Contents
* Notes about the generation of attention videos
* Reproduction of publication results



## Notes about the generation of attention videos
The generation of attention videos during training is done in a few steps:

1. during training, the attention matrices are extracted from the model at each ministep / epoch (configurable)
1. the matrices are stored serially in a `Zarr` array
1. after training, the `Zarr` arrays are re-processed to reorganize the storage structure for the quick recall of specific chemical compositions
1. the arrays are dynamically accessed and the attention matrices plotted using `matplotlib`
1. the plotted matrices are encoded into videos using `ffmpeg`

These steps require a large amount of fast storage and GPU VRAM. In addition, having a high number of CPU cores and system RAM will be helpful.
Alternatively, you can run the scripts on a high-performance computing cluster.



## Reproduction of publication results
To reproduce the publication results, run these scripts in order:

* `get_training_attention.py`: train CrabNet with different material property datasets and save Zarr arrays with the obtained attention values.
* `generate_attention_videos.py`: take the saved Zarr arrays, plot the attention maps and progress plots, and merge these into attention videos using `ffmpeg`.
* `guess_oxidation.py`: use Pymatgen to guess the oxidation of elements within the compounds in the material property datasets. Saved oxidation guesses are provided in the file `oxidation.zip` in the `data` directory.
* `get_crabnet_embeddings.py`: save learned element embeddings from CrabNet/HotCrab. Saved embeddings are provided in the files `embeddings_crabnet_mat2vec.zip` and `embeddings_crabnet_onehot.zip` in the `data` directory.
* `plot_element_correlation.py`: plot the Pearson correlation matrices between element vectors using different element representations (both static and interactive plots).
* `plot_local_edm_umap.py`: get individual EDM representations of atoms from within compounds and plot them as a UMAP.
* `plot_global_edm_umap.py`: get global EDM representations of compounds and plot them as a UMAP.
* `get_dataset_stats.py`: get descriptive statistics of the datasets as well as compute and plot Shannon entropy of the datasets.