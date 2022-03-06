# -*- coding: utf-8 -*-
"""
get umap embeddings!
"""
import sys
import os
import json

import gc
import glob
from PIL import Image
import subprocess

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import umap
from sklearn.manifold import TSNE

import torch
from torch import nn

from tqdm import tqdm
from time import time
from datetime import timedelta

import crabnet.kingcrab

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
from utils.attention_utils import collapse_edm
from utils.composition import generate_features

from sklearn.metrics import roc_auc_score

from utils.utils import CONSTANTS, get_obj_size, clear_cache, \
    linear, torch_memory_debug

from utils.oxidation_utils import find_oxidations, get_ionic_or_nonionic

if sys.platform == 'linux':
    from cuml import UMAP as cuUMAP
    from cuml import TSNE as cuTSNE

compute_device = get_compute_device(prefer_last=True)
proc_device = get_compute_device(force_cpu=False)

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

plt.rcParams['figure.facecolor'] = 'white'


# %%
data_dir = 'data/benchmark_data'
# mat_prop = 'aflow__ael_bulk_modulus_vrh'
mat_props = os.listdir(data_dir)

# mat_prop = 'aflow__ael_bulk_modulus_vrh' # compatible with onehot features
# mat_props = ['mp_bulk_modulus'] # compatible with mat2vec features
# mat_prop = 'CritExam__Ed'
# mat_prop = 'OQMD_Bandgap'
# mat_props = ['OQMD_Bandgap']
# mat_props = ['mp_bulk_modulus']
# mat_props = ['OQMD_Bandgap']
mat_props = ['OQMD_Bandgap', 'OQMD_Energy_per_atom', 'OQMD_Formation_Enthalpy', 'OQMD_Volume_per_atom']
# mat_props = [m for m in mat_props if 'OQMD' in m]


# %%
def get_model(mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = rf'data/benchmark_data/{mat_prop}/train.csv'
    val_data = rf'data/benchmark_data/{mat_prop}/val.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_benchmark__predictions'
    # save_path = 'publication_predictions/onehot_benchmark__predictions'
    # save_path = 'publication_predictions/random_200_benchmark__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(f'{mat_prop}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = rf'data/benchmark_data/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(mat_prop, classification, file_name, verbose=True):
    model = load_model(mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'\n{mat_prop} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'\n{mat_prop} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)
    return model, mae


# %%
class SaveFormulae:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.src = []
        self.frac = []
        self.x_emb = []
        self.acts = []
        self.preds = []
        self.counter = 0

    def __call__(self, module, module_in, module_out):
        # only capture output if requires_grad == False (i.e. in validation)
        # only capture output if requires_grad == True (i.e. in training)
        if model.capture_flag:
            src = module_in[0]
            frac = module_in[1]
            # indices representing atoms for each compound
            self.src.append(src.detach())
            self.frac.append(frac.detach())
            # embedded atoms for each compound
            self.x_emb.append(module.embed(src).detach() * \
                              2**module.emb_scaler.detach())
            # the module_out here is the EDM' tensor ((Q*KT)*V)
            self.outputs.append(module_out.detach())
            # self.acts.append(model.act_v)
            # self.preds.append(model.pred_v)
            self.counter += 1

    def clear(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.src = []
        self.frac = []
        self.acts = []
        self.preds = []
        self.counter = []


class SaveQKV:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = 0

    def __call__(self, module, module_in, module_out):
        # only capture output if requires_grad == False (i.e. in validation)
        # only capture output if requires_grad == True (i.e. in training)
        if model.capture_flag:
            # module_inputs are the "x" tensors
            self.inputs.append(*module_in)
            # weights are the attention in projection weights
            self.weights.append(module.self_attn.in_proj_weight.detach())
            # biases are the attention in projection biases
            self.biases.append(module.self_attn.in_proj_bias.detach())
            # plain-text full formula strings
            self.formulae.append(model.formula_current)
            # self.acts.append(model.act_v)
            # self.preds.append(model.pred_v)
            self.counter += 1

    def clear(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = []


# %%
for mat_prop in tqdm(mat_props, desc='Iter mat_prop'):
    print(f'{mat_prop = }')
    # TODO: set scale=False to avoid normalizing fractional amounts
    # instantiate CrabNet model that doesn't drop unary (pure) compounds
    model = Model(CrabNet(compute_device=compute_device,
                          residual_nn='roost')
                  .to(compute_device),
                  model_name=f'{mat_prop}',
                  verbose=True,
                  drop_unary=False,
                  scale=True)
    model.load_network(f'{mat_prop}.pth')

    file_name = 'test.csv'
    data = rf'data/benchmark_data/{mat_prop}/{file_name}'

    ox_file = f'data/oxidation/formulae+ox_{mat_prop}.json'
    ox_file_exists = os.path.exists(ox_file)
    if ox_file_exists:
        with open(ox_file, 'r') as f:
            ox_dict = json.load(f)

    # TODO: manually replace with all elements
    # data = 'data/all_elements.csv'

    model.load_data(data, batch_size=2**9, train=False)

    # set model capture flag to true
    model.capture_flag = True
    mod_list = [m for m in model.model.modules()]

    # %% get actual and predicted values from the model
    acts, preds, _, _ = model.predict(model.data_loader)

    # %%
    save_formulae = SaveFormulae()
    hook_handles_formulae = []

    for layer in model.model.modules():
        # if isinstance(layer, torch.nn.TransformerEncoderLayer):
        if isinstance(layer, crabnet.kingcrab.Encoder):
            print(f'hooked formulae')
            handle = layer.register_forward_hook(save_formulae)
            hook_handles_formulae.append(handle)

    # %%
    save_qkv = SaveQKV()
    hook_handles_qkv = []

    for layer in model.model.modules():
        if isinstance(layer, nn.TransformerEncoderLayer):
            print(f'hooked qkv')
            handle = layer.register_forward_hook(save_qkv)
            hook_handles_qkv.append(handle)

    # %% run model to populate hooks
    model, output = get_results(model)


    # %% set plotting parameters based on number of compounds
    n_compounds = len(acts)
    msize = 30
    malpha = 0.7
    min_dist = 0.2

    if n_compounds > 4000:
        malpha = 0.4
        min_dist = 0.4
    if n_compounds > 10_000:
        msize = 10
        malpha = 0.35
    if n_compounds > 50_000:
        msize = 3
        malpha = 0.3
    if n_compounds > 100_000:
        msize = 2
        malpha = 0.1
        min_dist = 0.5


    # %% work with the weights
    n_hooks = len(hook_handles_qkv)
    # specify the layer to grab
    layers = [0, 1, 2]

    # TODO change this later
    # layers = [layers[-1]]
    # layers = [0]

    for layer in tqdm(layers, desc='Iter layers'):
        assert layer <= n_hooks - 1 # one hook is registered per layer
        torch_memory_debug()

        # inputs has the dimension [n_el, bsz, d_model]
        n_el, bsz, d_model = save_qkv.inputs[layer].shape

        querys = save_qkv.inputs[layer::n_hooks]
        querys = torch.cat(querys, dim=1).detach().to(proc_device)

        clear_cache()

        full_formulae = save_qkv.formulae[layer::n_hooks]
        full_formulae = [item for ls in full_formulae for item in ls]
        full_formulae = np.array(full_formulae)


        # %%
        cons = CONSTANTS()
        elements = cons.atomic_symbols
        symbol_idx_dict = cons.symbol_idx_dict

        # %% work with the formulae
        # get UMAP embeddings of the Q, K, V of specific elements
        srcs = save_formulae.src
        srcs = torch.cat(srcs, dim=0).to(proc_device)

        fracs = save_formulae.frac
        fracs = torch.cat(fracs, dim=0).to(proc_device)

        data_dir_base = 'data/umap'
        fig_dir_base = 'figures/umap'

        # data_dir_base = 'data/t-sne'
        # fig_dir_base = 'figures/t-sne'
        os.makedirs(data_dir_base, exist_ok=True)
        os.makedirs(fig_dir_base, exist_ok=True)

        data_dir = f'{data_dir_base}/{mat_prop}'
        os.makedirs(data_dir, exist_ok=True)
        fig_dir = f'{fig_dir_base}/{mat_prop}'
        os.makedirs(fig_dir, exist_ok=True)

        n_neighbors = 15
        n_components = 2
        perplexity = 30
        # get n_elements for each compound
        n_elements = (srcs != 0).sum(1).cpu().numpy().astype('int32')

        # get the hidden state of the EDMs
        X = collapse_edm(querys, srcs, sum_feat=False)

        # subsample if too many compounds
        # n_samples = 5000
        # n_samples = X.shape[0]
        # idx_sample = np.random.choice(X.shape[0], size=(n_samples,), replace=False)
        # X = X[idx_sample]
        # n_elements = n_elements[idx_sample]
        # acts = acts[idx_sample]
        # preds = preds[idx_sample]
        # full_formulae = full_formulae[idx_sample]

        if sys.platform == 'win32':
            # working on CPU
            Xred = umap.UMAP(metric='cosine', n_neighbors=n_neighbors,
                              n_components=n_components,
                              min_dist=min_dist, random_state=42)
            Xemb = Xred.fit_transform(X)

            # red = TSNE(metric='euclidean', init='pca',
            #             perplexity=perplexity, square_distances=True,
            #             random_state=42)
            # Xemb = red.fit_transform(X)

        elif sys.platform == 'linux':
            # RAPIDS and cuML are installed, use CUDA UMAP
            Xred = cuUMAP(n_neighbors=n_neighbors,
                          n_components=n_components,
                          min_dist=min_dist)
            Xemb = Xred.fit_transform(X)

            # red = cuTSNE(perplexity=perplexity, square_distances=True,
            #             random_state=42)
            # Xemb = red.fit_transform(X)

        if n_components == 2:
            columns = ['formula', 'comp0', 'comp1', 'act', 'pred']

            # color by number of elements
            fig = plt.figure(figsize=(10, 8.5))
            fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=n_elements,
                              cmap='viridis', s=msize, alpha=malpha)
            plt.clim(n_elements.min(), n_elements.max())
            plt.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            cbar = plt.colorbar(boundaries=np.arange(n_elements.min()-1, n_elements.max()+1)+0.5)
            cbar.set_ticks(np.arange(n_elements.min(), n_elements.max()+1))
            cbar.set_alpha(1)
            cbar.set_label('number of elements in the compound')
            cbar.draw_all()
            # plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            plt.savefig(f'{fig_dir}/{mat_prop}_all_nelements_layer{layer}_UMAP_{n_components}d.png',
                        bbox_inches='tight', dpi=300)

            # color by actual value
            ymin = np.min([acts.min(), preds.min()])
            ymax = np.max([acts.max(), preds.max()])
            fig = plt.figure(figsize=(10, 8.5))
            fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=acts,
                              cmap='plasma', s=msize, alpha=malpha)
            plt.clim(ymin, ymax)
            plt.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            cbar = plt.colorbar(fig)
            cbar.set_alpha(1)
            cbar.set_label('target value')
            cbar.draw_all()
            # plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            plt.savefig(f'{fig_dir}/{mat_prop}_all_act_layer{layer}_UMAP_{n_components}d.png',
                        bbox_inches='tight', dpi=300)

            # color by predicted value
            fig = plt.figure(figsize=(10, 8.5))
            fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=acts,
                              cmap='plasma', s=msize, alpha=malpha)
            avg_preds = preds.mean()
            std_preds = preds.std()
            plt.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.clim(ymin, avg_preds + 6*std_preds)
            cbar = plt.colorbar(fig)
            cbar.set_alpha(1)
            cbar.set_label('predicted value')
            cbar.draw_all()
            # plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            plt.savefig(f'{fig_dir}/{mat_prop}_all_pred_layer{layer}_UMAP_{n_components}d.png',
                        bbox_inches='tight', dpi=300)

            # color by error (pred - actual)
            fig = plt.figure(figsize=(10, 8.5))
            error = preds - acts
            fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=error,
                              cmap='coolwarm', s=msize, alpha=malpha)
            avg_err = error.mean()
            std_err = error.std()
            plt.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.clim(avg_err - 2*std_err, avg_err + 2*std_err)
            cbar = plt.colorbar(fig)
            cbar.set_alpha(1)
            cbar.set_label('error (predicted – actual)')
            cbar.draw_all()
            # plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            plt.savefig(f'{fig_dir}/{mat_prop}_all_error_layer{layer}_UMAP_{n_components}d.png',
                        bbox_inches='tight', dpi=300)

            # color by std.dev of electronegativity (similar to EMD paper)
            print('Calculating electronegativity')
            df = pd.DataFrame({'formula': full_formulae, 'target': 0})
            X_ox, y, formulae, skipped = generate_features(df)
            X_ox.index = full_formulae
            EN_cols = ['Pauling_Electronegativity',
                       'MB_electonegativity',
                       'Gordy_electonegativity',
                       'Mulliken_EN',
                       'Allred-Rockow_electronegativity']
            electro_neg = X_ox['dev_Pauling_Electronegativity']
            avg_EN = electro_neg.mean()
            std_EN = electro_neg.std()
            fig = plt.figure(figsize=(10, 8.5))
            # Spectral and coolwarm are good color maps
            # alternatively: https://seaborn.pydata.org/generated/seaborn.diverging_palette.html
            cmap = plt.get_cmap('coolwarm').copy()
            # cmap.set_bad('lightgray')
            fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=electro_neg,
                              cmap=cmap, s=msize, alpha=malpha,
                              plotnonfinite=True)
            plt.tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False,
                            right=False,
                            left=False,
                            labelleft=False)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.clim(0, avg_EN + 2*std_EN)
            cbar = plt.colorbar(fig)
            cbar.set_alpha(1)
            cbar.set_label('std.dev electronegativity')
            cbar.draw_all()
            # plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            plt.savefig(f'{fig_dir}/{mat_prop}_all_EN_layer{layer}_UMAP_{n_components}d.png',
                        bbox_inches='tight', dpi=300)

            # color by whether oxidation or not
            # if ox_file_exists:
            #     atom_states = np.array(get_ionic_or_nonionic(ox_dict))
            #     # atom_states = atom_states[idx_sample]
            #     fig = plt.figure(figsize=(10, 8.5))
            #     # Spectral and coolwarm are good color maps
            #     # alternatively: https://seaborn.pydata.org/generated/seaborn.diverging_palette.html
            #     cmap = plt.get_cmap('coolwarm').copy()
            #     # cmap.set_bad('lightgray')
            #     fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=atom_states,
            #                       cmap=cmap, s=msize, alpha=malpha,
            #                       plotnonfinite=True)
            #     plt.clim(np.nanmin(atom_states), np.nanmax(atom_states))
            #     plt.tick_params(axis='both',
            #                     which='both',
            #                     bottom=False,
            #                     top=False,
            #                     labelbottom=False,
            #                     right=False,
            #                     left=False,
            #                     labelleft=False)
            #     plt.xlabel('UMAP 1')
            #     plt.ylabel('UMAP 2')
            #     cbar = plt.colorbar(fig)
            #     cbar.set_alpha(1)
            #     cbar.set_label('oxidation state')
            #     cbar.draw_all()
            #     plt.title(f'{mat_prop}\nlayer{layer} embeddings of all {len(X)} compounds in the {mat_prop} dataset')
            #     plt.savefig(f'{fig_dir}/{mat_prop}_all_oxidation_layer{layer}_UMAP_{n_components}d.png',
            #                 bbox_inches='tight', dpi=300)

        elif n_components == 3:
            columns = ['formula', 'comp0', 'comp1', 'comp2', 'act', 'pred']

        out = np.concatenate((full_formulae[:, np.newaxis],
                              Xemb,
                              acts[:, np.newaxis],
                              preds[:, np.newaxis]), axis=1)
        df = pd.DataFrame(out, columns=columns)

        if ox_file_exists:
            atom_states = np.array(get_ionic_or_nonionic(ox_dict))
            # atom_states = atom_states[idx_sample]
            df['oxidation'] = atom_states

        df.to_csv(f'{data_dir}/{mat_prop}_all_layer{layer}_UMAP_{n_components}d.csv', index=False)

        clear_cache()

        plt.close('all')

    # delete references and clear CUDA memory
    clear_cache()
    # clear_cache(save_qkv)
    # clear_cache(save_formulae)
    # torch_memory_debug()
