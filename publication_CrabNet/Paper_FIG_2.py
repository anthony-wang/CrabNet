import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
import seaborn as sns

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

import torch

from utils.utils import CONSTANTS

compute_device = get_compute_device()
cons = CONSTANTS()


# %%
mat_prop = 'aflow__Egap'
torchnet_params = {'d_model': 512, 'N': 3, 'heads': 4}
symbol_idx_dict = {val: key for key, val in cons.idx_symbol_dict.items()}

elem = 'Si'
# elem = 23
if type(elem) == int:
    elem_sym = cons.idx_symbol_dict[elem]
    elem_Z = elem
elif type(elem) == str:
    elem_Z = symbol_idx_dict[elem]
    elem_sym = elem

train_data1 = rf'data\benchmark_data\{mat_prop}\train.csv'
datas = [train_data1]
in_fracs = ['Average Attention (AFLOW Egap)']


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        module_out = [out.detach().cpu() for out in module_out]
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

save_output = SaveOutput()

for data, in_frac  in zip(datas, in_fracs):

    # Create a model
    model = Model(CrabNet(**torchnet_params, compute_device=compute_device).to(compute_device))
    model.load_network(f'{mat_prop}.pth')
    hook_handles = []

    # Insert forward hooks into model
    for layer in model.model.modules():
        if isinstance(layer, torch.nn.modules.activation.MultiheadAttention):
            # print('isinstance')
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

    model.load_data(data)  # data is reloaded to model.data_loader

    save_output.clear()
    output = model.predict(model.data_loader)

    elem_pred = [(i, out, output[1][i]) for
                 i, out in enumerate(output[2]) if elem_sym in out]
    df_elem = pd.DataFrame(elem_pred, columns=['idx', 'formula', 'prediction'])

    mod_out = save_output.outputs

    n_mats = len(mod_out)  # number of output matrices from hook
    bsz = model.data_loader.batch_size  # batch size from data loader
    B = len(model.data_loader)  # total number of batches from data loader
    H = model.model.heads  # number of heads
    N = model.model.N  # number of layers
    n_data = len(model.data_loader.dataset)
    n_elements = model.n_elements

    assert n_mats == N * B, 'something is wrong with the matrices'

    attn_data = torch.zeros(size=(n_data, N, H, n_elements, n_elements))
    for layer in range(N):
        sliceN = [save_output.outputs[i][1].unsqueeze(1) for
                  i in range(layer, n_mats, N)]
        sliceN = torch.cat(sliceN, dim=0)
        attn_data[:, layer:layer+1, :, :, :] = sliceN

    save_output.clear()   # free up CPU RAM after getting attn info
    attn_data = attn_data.detach().cpu().numpy()
    data_loader = model.data_loader



    def get_datum(data_loader, idx=0):
        datum = data_loader.dataset[idx]
        return datum

    def get_x(data_loader, idx=0):
        x = get_datum(data_loader, idx=idx)[0]
        return x

    def get_atomic_numbers(data_loader, idx=0):
        nums = get_x(data_loader, idx=idx).chunk(2)[0].detach().cpu().numpy()
        nums = nums.astype(int)
        return nums

    def get_atomic_fracs(data_loader, idx=0):
        nums = get_x(data_loader, idx=idx).chunk(2)[1].detach().cpu().numpy()
        return nums

    def get_target(data_loader, idx=0):
        target = get_datum(data_loader, idx=idx)[1].detach().cpu().numpy()
        return target

    def get_form(data_loader, idx=0):
        form = get_datum(data_loader, idx=idx)[2]
        return form


    def get_attention(attn_mat, idx=0, layer=0, head=0):
        """
        Get one slice of the attention map.

        Parameters
        ----------
        attn_mat : Tensor
            attn_mat is numpy array in the shape of [S, N, H, d, d], where
            S is the total number of data samples,
            N is the layer number in the attention mechanism,
            H is the head number in the attention mechanism, and
            d is the attention dimension in each head.
        idx : int, optional
            Index of the input material. The default is 0.
        layer : int, optional
            Layer number in the attention mechanism. The default is 0.
        head : int, optional
            Head number in the attention mechanism. The default is 0.

        Returns
        -------
        attn : Tensor

        """
        attn_mat = attn_mat
        assert len(attn_mat.shape) == 5, 'input attn_map is of the wrong shape'
        if head == 'average':
            attn = attn_mat[idx, layer, :, :, :]
            attn = np.mean(attn, axis=0)
        elif isinstance(head, int):
            attn = attn_mat[idx, layer, head, :, :]
        return attn


    attn_mat = attn_data.copy()

    data_loader = model.data_loader

    idx=1
    layer=0

    other_dict = {i: [] for i in range(1, 119)}

    option = [0, 1, 2, 3, 'average']
    option_texts = ['a)', 'b)', 'c)', 'd)', 'average']

    idx_plot = 0
    head_option = option[idx_plot]
    option_text = option_texts[idx_plot]

    for idx in range(len(data_loader.dataset)):
        map_data = get_attention(attn_mat, idx=idx, layer=layer, head=head_option)
        atom_fracs = get_atomic_fracs(data_loader, idx=idx)
        form = get_form(data_loader, idx=idx)
        atomic_numbers = get_atomic_numbers(data_loader, idx=idx).ravel().tolist()
        idx_symbol_dict = cons.idx_symbol_dict
        atoms = [idx_symbol_dict[num] for num in atomic_numbers]
        atom_presence = np.array(atom_fracs > 0)
        mask = atom_presence * atom_presence.T
        map_data = map_data * mask
        if elem_Z in atomic_numbers:
            row = atomic_numbers.index(elem_Z)
            for atomic_number in atomic_numbers:
                if atomic_number == 0:
                    continue
                col = atomic_numbers.index(atomic_number)
                # get the raw attention value
                other_dict[atomic_number].append(map_data[row, col])


    all_symbols = ['None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    property_tracker = {all_symbols[key]: np.array(val).mean() for key, val
                        in other_dict.items()
                        if len(val) != 0}

    def plot(mat_prop, property_tracker):
        ptable = pd.read_csv('data/element_properties/ptable.csv')
        ptable.index = ptable['symbol'].values
        elem_tracker = ptable['count']
        n_row = ptable['row'].max()
        n_column = ptable['column'].max()

        elem_tracker = elem_tracker + pd.Series(property_tracker)

        # log_scale = True
        log_scale = False

        fig, ax = plt.subplots(figsize=(n_column, n_row))
        rows = ptable['row']
        columns = ptable['column']
        symbols = ptable['symbol']
        rw = 0.9  # rectangle width (rw)
        rh = rw  # rectangle height (rh)
        for row, column, symbol in zip(rows, columns, symbols):
            row = ptable['row'].max() - row
            cmap = sns.cm.rocket_r
            count_min = elem_tracker.min()
            count_max = elem_tracker.max()
            count_min = 0
            count_max = 1
            norm = Normalize(vmin=count_min, vmax=count_max)
            count = elem_tracker[symbol]
            if log_scale:
                norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
                if count != 0:
                    count = np.log(count)
            color = cmap(norm(count))
            if np.isnan(count):
                color = 'silver'
            if row < 3:
                row += 0.5
            # element box
            rect = patches.Rectangle((column, row), rw, rh,
                                     linewidth=1.5,
                                     edgecolor='gray',
                                     facecolor=color,
                                     alpha=1)
            # plot element text
            text = plt.text(column+rw/2, row+rw/2, symbol,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=22,
                     fontweight='semibold', color='white')

            text.set_path_effects([path_effects.Stroke(linewidth=3,
                                                       foreground='#030303'),
                           path_effects.Normal()])

            ax.add_patch(rect)

        granularity = 20
        for i in range(granularity):
            value = (1-i/(granularity-1))*count_min + (i/(granularity-1)) * count_max
            if log_scale:
                if value != 0:
                    value = np.log(value)
            color = cmap(norm(value))
            length = 9
            x_offset = 3.5
            y_offset = 7.8
            x_loc = i/(granularity) * length + x_offset
            width = length / granularity
            height = 0.35
            rect = patches.Rectangle((x_loc, y_offset), width, height,
                                     linewidth=1.5,
                                     edgecolor='gray',
                                     facecolor=color,
                                     alpha=1)

            if i in [0, 4, 9, 14, 19]:
                text = f'{value:0.2f}'
                if log_scale:
                    text = f'{np.exp(value):0.1e}'.replace('+', '')
                plt.text(x_loc+width/2, y_offset-0.4, text,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontweight='semibold',
                         fontsize=20, color='k')

            ax.add_patch(rect)

        legend_title = f'{elem_sym}, {in_frac}'
        plt.text(x_offset+length/2, y_offset+0.7,
                 f'log({legend_title})' if log_scale else legend_title,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontweight='semibold',
                 fontsize=20, color='k')
        # add annotation for subfigure numbering
        plt.text(0.55, n_row+.1, option_text,
                 fontweight='semibold', fontsize=38, color='k')
        ax.set_ylim(-0.15, n_row+.1)
        ax.set_xlim(0.85, n_column+1.1)

        # fig.patch.set_visible(False)
        ax.axis('off')

        plt.draw()
        save_dir = 'figures/'
        if save_dir is not None:
            fig_name = f'{save_dir}/Figure2_{mat_prop}_ptable_{head_option}.png'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(fig_name, bbox_inches='tight', dpi=300)

        plt.pause(0.001)
        plt.close()

    plot(mat_prop, property_tracker)
