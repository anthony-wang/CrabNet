import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
from publication_CrabNet.benchmark_crabnet import get_results

import torch

from utils.utils import CONSTANTS

compute_device = get_compute_device()


# %%
mat_prop = 'mp_bulk_modulus'
crabnet_params = {'d_model': 512, 'N': 3, 'heads': 4}

model = Model(CrabNet(**crabnet_params, compute_device=compute_device).to(compute_device))
model.load_network(f'{mat_prop}.pth')

# Load the data you want to predict with
test_data = rf'data\benchmark_data\{mat_prop}\val.csv'
model.load_data(test_data)  # data is reloaded to model.data_loader
output = model.predict(model.data_loader)  # predict the data saved here


# %%
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

save_output = SaveOutput()
hook_handles = []

for layer in model.model.modules():
    if isinstance(layer, torch.nn.modules.activation.MultiheadAttention):
        # print('isinstance')
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

len(save_output.outputs)
model, _ = get_results(model)
len(save_output.outputs)


# %%
mod_out = save_output.outputs

len(mod_out)
len(mod_out[0])

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
    sliceN = [mod_out[i][1].unsqueeze(1) for i in range(layer, n_mats, N)]
    sliceN = torch.cat(sliceN, dim=0)
    attn_data[:, layer:layer+1, :, :, :] = sliceN

attn_data = attn_data.detach().cpu().numpy()
data_loader = model.data_loader
train_loader = model.train_loader


# %%
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


# %%
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
    attn = attn_mat[idx, layer, head, :, :]
    return attn


def plot_attention(map_data,
                   cbar_ax=None,
                   xlabel=None,
                   ylabel=None,
                   xticklabels=None,
                   yticklabels=None,
                   mask=True,
                   ax=None):
    "plots ONE attention map slice = map_data on given axis"
    if mask is not None:
        map_data = map_data * mask
    if xticklabels is None:
        xticklabels = list(range(map_data.shape[0]))
    if yticklabels is None:
        yticklabels = list(range(map_data.shape[0]))
    xticklabels = [f'{i:0.3g}' for i in xticklabels]

    import matplotlib as mpl
    cmap1 = mpl.colors.ListedColormap(['lightgray'])

    ax = sns.heatmap(map_data,
                     cmap=cmap1,
                     mask=map_data!=0,
                        linewidths=1,
                        linecolor='w',
                        cbar_ax=cbar_ax,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        annot=True,
                        fmt='.0f',
                        annot_kws={"size": 15},
                        vmin=0,
                        vmax=0,
                        ax=ax)

    ax = sns.heatmap(map_data,
                     cmap='rocket_r',
                     mask=map_data==0,
                        linewidths=1,
                        linecolor='gray',
                        cbar_ax=cbar_ax,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        annot=True,
                        fmt='.2f',
                        annot_kws={"size": 15},
                        vmin=0,
                        vmax=1,
                        ax=ax)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('center')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(yticklabels)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax.xaxis.set_tick_params(bottom=False, top=False, right=False,
                             left=False, labelbottom=True)
    ax2.xaxis.set_tick_params(bottom=False, top=False, right=False, left=False)
    ax.yaxis.set_tick_params(bottom=False, top=False, right=False,
                             left=False, labelleft=True)
    ax2.yaxis.set_tick_params(bottom=False, top=False, right=False, left=False)


    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(xticklabels)
    plt.setp(ax.get_xticklabels(), visible=True)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_all_heads(data_loader, attn_mat, idx=0, layer=0, mask=True):
    (S, N, H, d, d) = attn_mat.shape
    if H % 6 == 0:
        ncols = 6
    if H % 5 == 0:
        ncols = 5
    elif H % 4 == 0:
        if H == 4:
            ncols = 2
        else:
            ncols = 4
    elif H % 3 == 0:
        ncols = 3
    elif H % 2 == 0:
        ncols = 2
    else:
        ncols = 1
    nrows = H//ncols
    fig, fig_axes = plt.subplots(figsize=(4.5*ncols, 4.5*nrows),
                                 ncols=ncols, nrows=nrows,
                                 sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.33, wspace=0.25)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.03, 0.4])
    atom_fracs = get_atomic_fracs(data_loader, idx=idx)
    form = get_form(data_loader, idx=idx)
    atomic_numbers = get_atomic_numbers(data_loader, idx=idx).ravel().tolist()
    cons = CONSTANTS()
    idx_symbol_dict = cons.idx_symbol_dict
    atoms = [idx_symbol_dict[num] for num in atomic_numbers]

    if mask:
        atom_presence = np.array(atom_fracs > 0)
        mask = atom_presence * atom_presence.T

    plot_four = True
    label_abcd = ['a', 'b', 'c', 'd']
    if plot_four:
        for h in range(4):
            map_data = get_attention(attn_mat, idx=idx, layer=layer, head=h)
            n_el = 4
            plot_attention(map_data[:n_el, :n_el],
                           # xlabel='fractional amount',
                           # ylabel='atoms',
                           xticklabels=atom_fracs.ravel()[:n_el],
                           yticklabels=atoms[:n_el],
                           mask=mask[:n_el, :n_el],
                           ax=fig_axes.ravel()[h],
                           cbar_ax=cbar_ax,)
            fig_axes.ravel()[h].set_title(label=f'layer {layer}, head {h}')
            fig_axes.ravel()[h].set_title(label=f'{label_abcd[h]}){40*" "}',
                                          fontdict={'fontweight': 'bold'},
                                          y=1.05)
        plt.savefig('figures/Figure1_attention_plot_Al2O3.png',
                    bbox_inches='tight', dpi=300)
        plt.show()
        # exit
    else:
        for h in range(H):
            map_data = get_attention(attn_mat, idx=idx, layer=layer, head=h)
            plot_attention(map_data,
                           xlabel='fractional amount',
                           ylabel='atoms',
                           xticklabels=atom_fracs.ravel(),
                           yticklabels=atoms,
                           mask=mask,
                           ax=fig_axes.ravel()[h],
                           cbar_ax=cbar_ax,)
            fig_axes.ravel()[h].set_title(label=f'layer {layer}, head {h}')
        fig.suptitle(f'index: {idx}, formula: {form}')
        plt.savefig('figures/Figure1_attention_plot_Al2O3.png',
                    bbox_inches='tight', dpi=300)
        plt.show()
    return fig, fig_axes


# %%
formula_idx = [(i, out[2]) for i, out in enumerate(data_loader.dataset)]
find_formula = 'Al2O3'
idx = [i for i, out in enumerate(data_loader.dataset)
       if out[2] == find_formula]
if len(idx) > 0:
    print(f'Found {find_formula} at index {idx}')

for i in range(len(idx)):
    for n in range(N):
        plot_all_heads(data_loader, attn_data, idx=idx[i], layer=n)
        break
    break


