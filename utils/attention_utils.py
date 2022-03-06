import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

import torch

from tqdm import tqdm
import subprocess

from utils.get_compute_device import get_compute_device
from utils.utils import CONSTANTS

compute_device = get_compute_device(prefer_last=True)

plt.rcParams.update({'figure.max_open_warning': 50})



# %%
# get environment variables
ON_CLUSTER = os.environ.get('ON_CLUSTER')
HOME = os.environ.get('HOME')
USER_EMAIL = os.environ.get('USER_EMAIL')


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
def plot_progress(form, mat_prop, idx, act_data, pred_data, epoch, epochs, legend=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1,1,1)

    pred_progress = pred_data[idx, :epoch]
    target = act_data[idx, -1]

    epochs_bar = np.arange(epoch)

    ymin = np.nanmin(pred_data[idx, :])
    ymax = np.nanmax(pred_data[idx, :])

    if target > ymax:
        ymax = target
    if target < ymin:
        ymin = target
    yrange = ymax - ymin

    pred_plot = ax.plot(epochs_bar, pred_progress, 'bo-', alpha=0.5, label='pred')
    target_plot = ax.plot(epochs-1, target, 'rx', ms=10, label='target')
    target_line = ax.plot([0, epochs-1], [target, target], 'r--')

    n_digits = len(str(epochs))

    plt.xlabel('steps')
    plt.ylabel(f'{mat_prop}')
    plt.title(f'#{idx}: {form}, step {epoch:0{n_digits}d}/{epochs}')
    plt.ylim((ymin - 0.1*yrange, ymax + 0.1*yrange))
    if legend:
        plt.legend()

    return fig, ax, pred_plot, target_plot, target_line


# %%
def plot_progress_save(form, mat_prop, fig, ax, pred_plot, target_plot, target_line, idx, act_data, pred_data, epoch, epochs, name):
    if ((fig is None)
        or (ax is None)
        or (pred_plot is None)
        or (target_plot is None)
        or (target_line is None)):
            print('something was wrong with the object handles in plot_progress_save')
            fig, ax, pred_plot, target_plot, target_line = plot_progress(form, mat_prop, idx, act_data, pred_data, epoch, epochs)
    else:
        fig, ax, pred_plot, target_plot, target_line = redraw_progress(form, mat_prop, fig, ax, pred_plot, target_plot, target_line, idx, act_data, pred_data, epoch=epoch, epochs=epochs)
        fig.canvas.draw()

    if name is not None:
        fig.savefig(f'{name}',
                    bbox_inches='tight', dpi=100)

    return fig, ax, pred_plot, target_plot, target_line


# %%
def redraw_progress(form, mat_prop, fig, ax, pred_plot, target_plot, target_line, idx, act_data, pred_data, epoch, epochs):
    # source: https://stackoverflow.com/questions/59816675/is-it-possible-to-update-inline-plots-in-python-spyder
    pred_progress = pred_data[idx, :epoch]
    target = act_data[idx, -1]

    epochs_bar = np.arange(epoch)

    ymin = np.nanmin(pred_data[idx, :])
    ymax = np.nanmax(pred_data[idx, :])

    if target > ymax:
        ymax = target
    if target < ymin:
        ymin = target
    yrange = ymax - ymin

    pred_plot[0].set_data(epochs_bar, pred_progress)

    n_digits = len(str(epochs))

    plt.title(f'#{idx}: {form}, step {epoch:0{n_digits}d}/{epochs}')

    return fig, ax, pred_plot, target_plot, target_line


# %%
def get_attention(attn_mats, stride, idx=0, epoch=0, layer=0, head=0):
    """
    Get one slice of the attention map.

    Parameters
    ----------
    attn_mat : Tensor
        attn_mat is a list of numpy memmap arrays
            in the shape of [S, E, H, d, d], where
        S is the total number of data samples,
        L is the layer number (here is is always 1, since each layer is stored
                               in its own array file),
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
    # TODO: redo doc string above

    # get the actual idx location of the sample
    # (since the data repeats itself after 'stride' number of samples)
    act_idx = (idx * stride) + epoch

    attn_mat = attn_mats[layer]
    attn = attn_mat[act_idx, 0, head, :, :]
    return attn


# %%
# data_save_path = r'data_save\aflow__ael_bulk_modulus_vrh'
# files = [f'{data_save_path}/attn_data_layer{L}.npy' for L in range(3)]
# stride = 727

# attn_mats1 = [np.load(file) for file in files]
# attn_mats2 = [np.load(file, mmap_mode='r') for file in files]

# %timeit get_attention(attn_mats1, stride, idx=726, epoch=1, layer=1, head=0)
# %timeit get_attention(attn_mats2, stride, idx=726, epoch=1, layer=1, head=0)

# for mat in attn_mats2:
#     del mat
# del attn_mats2
# gc.collect()


# %%
def plot_attention(map_data,
                   xlabel=None,
                   ylabel=None,
                   xticklabels=None,
                   yticklabels=None,
                   mask=None,
                   annot=False,
                   ax=None,
                   cbar_ax=None,
                   cmap=None):

    if xticklabels is None:
        xticklabels = list(range(map_data.shape[0]))
    if yticklabels is None:
        yticklabels = list(range(map_data.shape[0]))
    # format xticklabels
    xticklabels = [f'{i:.3g}'.replace('0.', '.', 1) for i in xticklabels]
    n_el = len(xticklabels)

    if cmap is None:
        cmap = copy(plt.cm.magma_r)
        cmap.set_bad(color='lightgray')

    if mask is None:
        # calculate mask ourselves
        # val spots are spots holding values
        # empty spots are empty
        val_spots = np.ma.masked_where(map_data == 0, map_data)
        val_spots_mask = (val_spots.mask) + (val_spots.mask).T
        val_spots.mask = val_spots_mask
    else:
        map_data = map_data * mask
        val_spots = np.ma.masked_where(map_data == 0, map_data)

    # plot the heatmap
    im = ax.imshow(val_spots,
                   vmin=0, vmax=1.0,
                   aspect='equal',
                   interpolation='none',
                   cmap=cmap)

    if annot:
        # annotate the actual values of the cells
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                val = map_data[i, j]
                if val != 0:
                    text = ax.text(j, i, f'{val:0.2f}',
                                   ha="center", va="center",
                                   color="w", size=13)

    # x axis label at the bottom of the heatmap for fractional amount
    ax.set_xticks(np.arange(n_el))
    ax.set_yticks(np.arange(n_el))

    # set x axis label and rotate the y axis labels
    # rotate the fractional labels by 45 deg
    # ax.set_xticklabels(xticklabels, rotation=0, fontsize=15)
    ax.set_xticklabels(xticklabels, rotation=45, fontsize=14)

    ax.set_yticklabels(yticklabels, rotation=45, fontsize=14)

    ax.xaxis.set_tick_params(pad=-2)
    ax.yaxis.set_tick_params(pad=-2)

    # create white grid on the minor grids
    ax.set_xticks(np.arange(n_el)-0.5, minor=True)
    ax.set_yticks(np.arange(n_el)-0.5, minor=True)
    ax.grid(which="minor", color='darkgray', linestyle='-', linewidth=2)

    # stagger the x axis fractional amount labels
    # for tick in ax.xaxis.get_major_ticks()[::2]:
    #     tick.set_pad(-2)
    # for tick in ax.xaxis.get_major_ticks()[1::2]:
    #     tick.set_pad(10)

    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_verticalalignment('center')

    # x axis label at the top of the heatmap
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(yticklabels, rotation=45, fontsize=14)
    ax2.xaxis.set_tick_params(pad=-2)

    ax.tick_params(which="minor", bottom=False, left=False)
    ax2.tick_params(which="minor", bottom=False, left=False)

    # Turn spines off for both axes
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    for edge, spine in ax2.spines.items():
        spine.set_visible(False)

    # Turn ticks off
    ax.xaxis.set_tick_params(bottom=False, top=False,
                             right=False, left=False, labelbottom=True)
    ax.yaxis.set_tick_params(bottom=False, top=False,
                             right=False, left=False, labelleft=True)
    ax2.xaxis.set_tick_params(bottom=False, top=False, right=False, left=False)
    ax2.yaxis.set_tick_params(bottom=False, top=False, right=False, left=False)

    # set aspect ratio for both axes
    ax.set_box_aspect(1)
    ax2.set_box_aspect(1)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if cbar_ax is None:
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.tight_layout()

    return im, ax, ax2

# idx=128
# e=0
# layer=0
# fig, ims, axs, ax2s = plot_all_heads(data_loader, attn_data, idx=idx, epoch=e, layer=layer)
# fig


# %%
def plot_all_heads(data_loader, attn_mat, stride, idx=0, epoch=0, layer=0, mask=True):
    # we have four attention heads, so plot on 2 columns x 2 rows
    ncols = 2
    nrows = 2

    fig, axs = plt.subplots(figsize=(3.0*ncols, 3.0*nrows),
                                 ncols=ncols, nrows=nrows,
                                 sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.03, 0.4])
    atom_fracs = get_atomic_fracs(data_loader, idx=idx)
    form = get_form(data_loader, idx=idx)
    atomic_numbers = get_atomic_numbers(data_loader, idx=idx).ravel().tolist()
    cons = CONSTANTS()
    idx_symbol_dict = cons.idx_symbol_dict
    atoms = [idx_symbol_dict[num] for num in atomic_numbers]
    atoms = ['--' if x == 'None' else x for x in atoms]

    if mask:
        atom_presence = np.array(atom_fracs > 0)
        mask = atom_presence * atom_presence.T

    label_abcd = ['a', 'b', 'c', 'd']
    ims = []
    ax2s = []
    cmap = copy(plt.cm.magma_r)
    cmap.set_bad(color='lightgray')

    # we have four heads
    for h in range(4):
        map_data = get_attention(attn_mat,
                                 stride,
                                 idx=idx,
                                 epoch=epoch,
                                 layer=layer,
                                 head=h)
        n_el = attn_mat[0].shape[-1]
        im, ax, ax2 = plot_attention(map_data[:n_el, :n_el],
                            # xlabel='fractional amount',
                            # ylabel='atoms',
                            xticklabels=atom_fracs.ravel()[:n_el],
                            yticklabels=atoms[:n_el],
                            mask=mask[:n_el, :n_el],
                            ax=axs.ravel()[h],
                            cbar_ax=cbar_ax,
                            cmap=cmap)
        # axs.ravel()[h].set_title(label=f'layer {layer}, head {h}',
        #                               fontsize=14)
        ims.append(im)
        ax2s.append(ax2)
    fig.colorbar(im, cax=cbar_ax)
    # fig.suptitle(f'#{idx}: {form}, layer {layer}, heads 1-4')
    # axs.ravel()[h].set_title(label=f'{label_abcd[h]}){40*" "}',
    #                               fontdict={'fontweight': 'bold'},
    #                               y=1.05)

    return fig, ims, axs, ax2s


# %%
def plot_all_heads_save(data_loader, attn_mat, stride, fig, ims, axs, ax2s, idx=0, epoch=0, layer=0, mask=True, name=None):
    if ((fig is None)
        or (ims is None)
        or (axs is None)
        or (ax2s is None)):
            print('something was wrong with the object handles in plot_all_heads_save')
            fig, ims, axs, ax2s = plot_all_heads(data_loader, attn_mat, stride, idx=idx, epoch=epoch, layer=layer, mask=mask)
    else:
        fig, ims, axs, ax2s = redraw_all_heads(data_loader, attn_mat, stride, fig, ims, axs, ax2s, idx=idx, epoch=epoch, layer=layer)
        fig.canvas.draw()

    if name is not None:
        fig.savefig(f'{name}',
                    bbox_inches='tight', dpi=100)

    return fig, ims, axs, ax2s


# %%
def redraw_all_heads(data_loader, attn_mat, stride, fig, ims, axs, ax2s, idx=0, epoch=0, layer=0, mask=True):
    # source: https://stackoverflow.com/questions/59816675/is-it-possible-to-update-inline-plots-in-python-spyder
    # https://www.scivision.dev/fast-update-matplotlib-plots/
    # https://stackoverflow.com/a/43885275/1390489
    atom_fracs = get_atomic_fracs(data_loader, idx=idx)
    form = get_form(data_loader, idx=idx)
    atomic_numbers = get_atomic_numbers(data_loader, idx=idx).ravel().tolist()
    cons = CONSTANTS()
    idx_symbol_dict = cons.idx_symbol_dict
    atoms = [idx_symbol_dict[num] for num in atomic_numbers]
    atoms = ['--' if x == 'None' else x for x in atoms]

    if mask:
        atom_presence = np.array(atom_fracs > 0)
        mask = atom_presence * atom_presence.T

    label_abcd = ['a', 'b', 'c', 'd']

    # we have four attention heads
    for h in range(4):
        map_data = get_attention(attn_mat,
                                 stride,
                                 idx=idx,
                                 epoch=epoch,
                                 layer=layer,
                                 head=h)
        n_el = mask[0,:].sum()
        map_data = map_data * mask
        val_spots = np.ma.masked_where(map_data == 0, map_data)

        # update map data
        ims[h].set_array(val_spots)

    # update axs xticklabels
    xticklabels = atom_fracs.ravel()
    xticklabels = [f'{i:.3g}'.replace('0.', '.', 1) for i in xticklabels]
    axs.ravel()[0].set_xticklabels(xticklabels)

    # update axs yticklabels
    yticklabels = ['--' if x == 'None' else x for x in atoms]
    axs.ravel()[0].set_yticklabels(yticklabels)

    for axi in ax2s:
        # update ax2 xticklabels
        axi.set_xticklabels(yticklabels)

    fig.suptitle(f'#{idx}: {form}, layer {layer}, heads 1-4')
    # fig.canvas.draw()

    return fig, ims, axs, ax2s


# %%
def imgs_to_video(infile, outfile, encoder):
    ret = subprocess.run([
            'ffmpeg', '-y', '-r', '10',
            '-i', infile,
            '-c:v', encoder,
            '-tune', 'stillimage',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-vf', "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            '-movflags', '+faststart',
            outfile,
            ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )

    return ret


# %%
def vids_to_video(infile1, infile2, outfile, encoder):
    ret = subprocess.run([
            'ffmpeg',
            '-y',
            '-i', infile1,
            '-i', infile2,
            '-c:v', encoder,
            '-tune', 'stillimage',
            '-crf', '18',
            '-filter_complex', "[0]scale=550:550:force_original_aspect_ratio=decrease:flags=lanczos,pad=550:550:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1[0v];[1]scale=550:550:force_original_aspect_ratio=decrease:flags=lanczos,pad=550:550:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1[1v];[0v][1v]hstack,format=yuv420p",
            '-movflags', '+faststart',
            outfile,
            ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )

    return ret


# %%
def collapse_edm(edm, srcs, sum_feat=False):
    seq_len, n_compounds, d_model = edm.shape

    if isinstance(edm, torch.Tensor):
        edm = edm.detach().cpu().numpy()

    if isinstance(srcs, torch.Tensor):
        srcs = srcs.detach().cpu().numpy()

    if sum_feat:
        d_feats = 6 * d_model
    else:
        d_feats = 5 * d_model
    out_shape = (n_compounds, d_feats)

    sum_feats = []
    avg_feats = []
    range_feats = []
    var_feats = []
    dev_feats = []
    max_feats = []
    min_feats = []
    mode_feats = []
    targets = []
    formulas = []

    for c in tqdm(range(n_compounds), desc='Assigning Features...'):
        comp_mat = edm[:, c, :]
        range_feats.append(np.ptp(comp_mat, axis=0))
        var_feats.append(comp_mat.var(axis=0))
        max_feats.append(comp_mat.max(axis=0))
        min_feats.append(comp_mat.min(axis=0))
        # this might be wrong?
        avg_feats.append(comp_mat.sum(axis=0) / (srcs[c] != 0).sum(axis=0))

        # comp_frac_mat = comp_mat.T * frac_mat[h]
        # comp_frac_mat = comp_frac_mat.T
        # avg_feats.append(comp_frac_mat.sum(axis=0))

        if sum_feat:
            pass
            # sum_feats.append(comp_sum_mat.sum(axis=0))

    if sum_feat:
        conc_list = [sum_feats, avg_feats,
                     range_feats, var_feats, max_feats, min_feats]
        feats = np.concatenate(conc_list, axis=1)
    else:
        conc_list = [avg_feats,
                     range_feats, var_feats, max_feats, min_feats]
        feats = np.concatenate(conc_list, axis=1)

    return feats


# %%
if __name__ == '__main__':
    pass
