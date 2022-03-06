import sys
import os
import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import umap

import torch
from torch import nn

from tqdm import tqdm

import crabnet.kingcrab

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

from sklearn.metrics import roc_auc_score

from utils.utils import CONSTANTS, clear_cache, linear, torch_memory_debug
from utils.oxidation_utils import find_oxidations

if sys.platform == 'linux':
    from cuml import UMAP as cuUMAP

compute_device = get_compute_device(prefer_last=True)
proc_device = get_compute_device(force_cpu=False)

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

plt.rcParams['figure.facecolor'] = 'white'


# %%
data_dir = 'data/benchmark_data'
mat_props = os.listdir(data_dir)

mat_props = ['OQMD_Bandgap']


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
elem_ox = pd.read_json('data/oxidation/element_ox_states.json')
elem_ox.index = elem_ox['element']


# %%
for mat_prop in tqdm(mat_props, desc='Iter mat_prop'):
    print(f'{mat_prop = }')
    # Set scale=False to avoid normalizing fractional amounts
    # and instantiate CrabNet model that doesn't drop unary (pure) compounds
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

    model.load_data(data, batch_size=2**14, train=False)

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
            print('hooked formulae')
            handle = layer.register_forward_hook(save_formulae)
            hook_handles_formulae.append(handle)

    # %%
    save_qkv = SaveQKV()
    hook_handles_qkv = []

    for layer in model.model.modules():
        if isinstance(layer, nn.TransformerEncoderLayer):
            print('hooked qkv')
            handle = layer.register_forward_hook(save_qkv)
            hook_handles_qkv.append(handle)

    # %% run model to populate hooks
    model, output = get_results(model)

    # %% work with the weights
    n_hooks = len(hook_handles_qkv)
    # specify the layer to grab
    layers = [0, 1, 2]

    for layer in tqdm(layers, desc='Iter layers'):
        assert layer <= n_hooks - 1 # one hook is registered per layer
        torch_memory_debug()

        # inputs has the dimension [n_el, bsz, d_model]
        n_el, bsz, d_model = save_qkv.inputs[layer].shape

        weights = save_qkv.weights[layer]
        biases = save_qkv.biases[layer]
        querys = save_qkv.inputs[layer::n_hooks]

        # weights = torch.cat(weights, dim=0).detach().to(proc_device)
        # biases = torch.cat(biases, dim=0).detach().to(proc_device)
        querys = torch.cat(querys, dim=1).detach().to(proc_device)

        clear_cache()

        full_formulae = save_qkv.formulae[layer::n_hooks]
        full_formulae = [item for ls in full_formulae for item in ls]
        full_formulae = np.array(full_formulae)

        # %%
        Q, K, V = linear(querys, weights, biases).chunk(3, dim=-1)
        clear_cache()

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
        os.makedirs(data_dir_base, exist_ok=True)
        os.makedirs(fig_dir_base, exist_ok=True)

        atoms = ['Al', 'Fe', 'O', 'C', 'B', 'N', 'Ti', 'Cu', 'Li', 'W', 'Na',
                 'Si', 'Ag', 'Te', 'As', 'Cr', 'Pt', 'Tl', 'Sb']

        for atom_name in tqdm(atoms, desc='Iter atoms'):
            data_dir = f'{data_dir_base}/{mat_prop}'
            os.makedirs(data_dir, exist_ok=True)
            fig_dir = f'{fig_dir_base}/{mat_prop}'
            os.makedirs(fig_dir, exist_ok=True)

            atom = symbol_idx_dict[atom_name]
            atom = torch.tensor(atom, dtype=torch.int32).to(proc_device)

            findall = (srcs == atom)
            findall_cpu = findall.cpu()
            n_formulae = torch.any(findall, dim=1).sum()

            n_neighbors = 5
            n_components = 2
            min_dist = 0.3
            if layer == 0:
                min_dist = 0.8
            perplexity = 10

            if n_formulae < 2 * n_neighbors:
                print(f'\nn_formulae ({n_formulae}) < '
                      f'2*n_neighbors ({2*n_neighbors}), skipping {atom_name}')
                continue

            X = V.transpose(0,1)[findall]
            fracsx = fracs[findall]
            actsx = acts[torch.any(findall_cpu, dim=1)]
            predsx = preds[torch.any(findall_cpu, dim=1)]
            ymax = max(actsx.max(), predsx.max())
            ymin = max(actsx.min(), predsx.min())

            print(f'{n_formulae} formulae found with the element {atom_name}')

            formulae = full_formulae[findall_cpu.any(dim=1)]

            if sys.platform == 'win32':
                # working on CPU
                X = X.cpu().numpy()
                fracsx = fracsx.cpu().numpy()
                Xred = umap.UMAP(metric='euclidean', n_neighbors=n_neighbors,
                                  n_components=n_components,
                                  min_dist=min_dist, random_state=42)
                Xemb = Xred.fit_transform(X)

            elif sys.platform == 'linux':
                # assume RAPIDS and cuML are installed, use CUDA UMAP
                Xred = cuUMAP(n_neighbors=n_neighbors,
                              n_components=n_components,
                              min_dist=min_dist, random_state=42)
                Xemb = Xred.fit_transform(X)

            if n_components == 2:
                columns = ['formula', 'comp0', 'comp1', 'frac', 'act', 'pred']

                # color by fractional amount
                fig = plt.figure(figsize=(10, 8.5))
                fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=fracsx,
                                  cmap='viridis', s=50, alpha=0.15)
                plt.clim(0, 1.0)
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
                cbar.set_label('fractional amount of atom')
                cbar.draw_all()
                fig_path = f'{fig_dir}/{mat_prop}_{atom_name}_frac_layer{layer}_UMAP_{n_components}d.png'
                html_path = f'{fig_dir}/{mat_prop}_{atom_name}_frac_layer{layer}_UMAP_{n_components}d.html'
                plt.savefig(fig_path, bbox_inches='tight', dpi=200)
                fig_int = px.scatter(x=Xemb[:,0], y=Xemb[:,1],
                                     color=fracsx,
                                     hover_name=formulae,
                                     range_color=[0, 1.0],
                                     color_continuous_scale='Viridis')
                fig_int.update_layout(coloraxis_colorbar=dict(title='fractional amount of atom'))
                fig_int.write_html(html_path, include_plotlyjs='cdn')

                # color by target value
                fig = plt.figure(figsize=(10, 8.5))
                fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=actsx,
                                  cmap='plasma', s=50, alpha=0.15)
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
                fig_path = f'{fig_dir}/{mat_prop}_{atom_name}_act_layer{layer}_UMAP_{n_components}d.png'
                html_path = f'{fig_dir}/{mat_prop}_{atom_name}_act_layer{layer}_UMAP_{n_components}d.html'
                plt.savefig(fig_path, bbox_inches='tight', dpi=200)
                fig_int = px.scatter(x=Xemb[:,0], y=Xemb[:,1],
                                     color=actsx,
                                     hover_name=formulae,
                                     range_color=[ymin, ymax],
                                     color_continuous_scale='plasma')
                fig_int.update_layout(coloraxis_colorbar=dict(title='target value'))
                fig_int.write_html(html_path, include_plotlyjs='cdn')

                # color by predicted value
                fig = plt.figure(figsize=(10, 8.5))
                fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=predsx,
                                  cmap='plasma', s=50, alpha=0.15)
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
                cbar.set_label('predicted value')
                cbar.draw_all()
                fig_path = f'{fig_dir}/{mat_prop}_{atom_name}_pred_layer{layer}_UMAP_{n_components}d.png'
                html_path = f'{fig_dir}/{mat_prop}_{atom_name}_pred_layer{layer}_UMAP_{n_components}d.html'
                plt.savefig(fig_path, bbox_inches='tight', dpi=200)
                fig_int = px.scatter(x=Xemb[:,0], y=Xemb[:,1],
                                     color=predsx,
                                     hover_name=formulae,
                                     range_color=[ymin, ymax],
                                     color_continuous_scale='plasma')
                fig_int.update_layout(coloraxis_colorbar=dict(title='predicted value'))
                fig_int.write_html(html_path, include_plotlyjs='cdn')

                if ox_file_exists:
                    # color by estimated oxidation state
                    n_total, n_guesses, n_atoms, atom_states = find_oxidations(atom_name, ox_dict)
                    atom_states = np.asarray(atom_states, dtype='float32')
                    fig = plt.figure(figsize=(10, 8.5))
                    sel_atom_states = atom_states[findall_cpu.any(dim=1)]
                    # Spectral and coolwarm are good color maps
                    # alternatively: https://seaborn.pydata.org/generated/seaborn.diverging_palette.html
                    cmap = sns.diverging_palette(250, 15, s=75, l=60,
                                                  n=10, center="dark", as_cmap=True)
                    # cmap = plt.get_cmap('coolwarm').copy()
                    # cmap.set_bad('lightgrey')
                    no_ox_idx = np.isnan(sel_atom_states)
                    # plot invalid points (gray)
                    plt.scatter(Xemb[no_ox_idx,0], Xemb[no_ox_idx,1],
                                      c='lightgrey', s=50, alpha=0.2,
                                      plotnonfinite=True)
                    # plot points with estimated oxidation
                    fig = plt.scatter(Xemb[:,0], Xemb[:,1], c=sel_atom_states,
                                      cmap=cmap, s=50, alpha=0.4,
                                      plotnonfinite=False)

                    ox_type = 'icsd_oxidation_states'
                    ox_states = np.array(elem_ox.loc[atom_name, ox_type])
                    ox_min = np.nanmin(np.concatenate([sel_atom_states, ox_states]))
                    ox_max = np.nanmax(np.concatenate([sel_atom_states, ox_states]))
                    plt.clim(ox_min, ox_max)
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
                    cbar.set_label('oxidation state')
                    cbar.draw_all()
                    fig_path = f'{fig_dir}/{mat_prop}_{atom_name}_oxidation_layer{layer}_UMAP_{n_components}d.png'
                    html_path = f'{fig_dir}/{mat_prop}_{atom_name}_oxidation_layer{layer}_UMAP_{n_components}d.html'
                    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
                    fig_int1 = px.scatter(x=Xemb[no_ox_idx,0], y=Xemb[no_ox_idx,1],
                                         hover_name=formulae[no_ox_idx],
                                         opacity=0.3,
                                         symbol_sequence=['x' for pt in no_ox_idx if pt])
                    fig_int2 = px.scatter(x=Xemb[~no_ox_idx,0], y=Xemb[~no_ox_idx,1],
                                         color=sel_atom_states[~no_ox_idx],
                                         hover_name=formulae[~no_ox_idx],
                                         range_color=[ox_min, ox_max],
                                         color_continuous_scale='balance')
                    fig_int = go.Figure(data=fig_int1.data + fig_int2.data)
                    fig_int.update_layout(coloraxis_colorbar=dict(title='oxidation state'))
                    fig_int.write_html(html_path, include_plotlyjs='cdn')

            elif n_components == 3:
                columns = ['formula', 'comp0', 'comp1', 'comp2', 'frac', 'act', 'pred']

            out = np.concatenate((formulae[:,np.newaxis],
                                  Xemb,
                                  fracsx[:,np.newaxis],
                                  actsx[:,np.newaxis],
                                  predsx[:,np.newaxis]), axis=1)
            df = pd.DataFrame(out, columns=columns)
            df.to_csv(f'{data_dir}/{mat_prop}_{atom_name}_layer{layer}_UMAP_{n_components}d.csv', index=False)

            clear_cache()
            plt.close('all')

        clear_cache((Q, K, V, weights, biases, querys))
        plt.close('all')

