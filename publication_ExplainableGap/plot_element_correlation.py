import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px

from tqdm import tqdm

from utils.utils import CONSTANTS


# %%
# set correct Seaborn plotting context and increase font size
sns.set_context('paper', font_scale=2)


# %%
cons = CONSTANTS()
all_elements = cons.atomic_symbols[1:]
oliynyk_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
                   'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                   'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                   'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                   'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                   'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                   'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                   'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U']
symbol_idx_dict = cons.symbol_idx_dict


# %%
cbfvs = ['oliynyk', 'magpie', 'jarvis', 'mat2vec']
data_dir = [f'data/element_properties/{cbfv}.csv' for cbfv in cbfvs]

data_dir = 'data/embeddings_crabnet_mat2vec'
# data_dir = 'data/embeddings_crabnet_onehot'


if isinstance(data_dir, list):
    mat_props = data_dir
elif os.path.isdir(data_dir):
    mat_props = os.listdir(data_dir)
else:
    mat_props = [data_dir]

fig_dir = 'figures/correlation_atoms/'
os.makedirs(fig_dir, exist_ok=True)


# %%
def get_important_labels(arr):
    # hand-picking some important atomic numbers
    important_zs = [1, 3, 5, 11, 13, 19, 21, 25, 31, 37, 43, 49, 55, 57, 71,
                    75, 81, 87, 92, 98, 107, 113, 118]

    # use a constant scale for atomic numbers
    important_zs = np.arange(1, 119, 5)

    return important_zs

# %%
for mat_prop in tqdm(mat_props):
    print(mat_prop)
    if os.path.isfile(mat_prop):
        name = os.path.basename(mat_prop).split('.csv')[0]
        path = mat_prop
    elif os.path.isdir(data_dir):
        name = mat_prop.split('.csv')[0]
        path = os.path.join(data_dir, f'{name}.csv')
    else:
        path = data_dir
        name = mat_prop.split('/')[-1].split('.csv')[0]

    df = pd.read_csv(path)

    if 'element' not in df.columns:
        df['element'] = df.iloc[:,0]
        df = df.drop(columns=['Unnamed: 0'])

    # get rid of 'Null' element representation
    null_row = df['element'] == 'Null'
    df = df[~null_row]

    # grab only the elements that are in oliynyk
    df.index = df['element']
    # df = df.loc[oliynyk_elements]

    elements = df.pop('element')
    element_idxs = [symbol_idx_dict[el] for el in elements]


    # %%
    # normalize the dataframe
    # zero mean, unit variance
    df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)


    # %%
    cmap = sns.diverging_palette(20, 220, n=200, center='light')
    cbar_kws = dict(shrink=0.8)
    fontsize = 100 / np.sqrt(df.shape[0])


    # %%
    # set the plot to be cropped similarly to Oliynyk or not (missing atoms);
    # this improves the ease of comparison between the plots of different
    # properties, however, it can potentially hide data
    for plot_like_oliynyk in [False, True]:
        max_z = 118
        min_z = 92
        current_z = max(element_idxs)

        if plot_like_oliynyk:
            current_elements = all_elements[:min_z]
            missing_elements = [el for el in current_elements if el not in elements]
            fig_path = os.path.join(fig_dir, f'{name}_corr_atoms_truncated.png')
            html_path = os.path.join(fig_dir, f'{name}_corr_atoms_truncated.html')
        else:
            current_elements = all_elements[:current_z]
            missing_elements = [el for el in current_elements if el not in elements]
            fig_path = os.path.join(fig_dir, f'{name}_corr_atoms.png')
            html_path = os.path.join(fig_dir, f'{name}_corr_atoms.html')

        df_extended = df.reindex(current_elements)

        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()

        mask = np.triu(np.ones_like(corr2, dtype=bool))

        arr = range(1, len(current_elements)+1)
        z_ticks = get_important_labels(arr)
        if name == 'megnet16':
            y_ticks = [current_elements[num] if num in z_ticks else '' for num in arr]
        else:
            y_ticks = [current_elements[num - 1] if num in z_ticks else '' for num in arr]
        z_ticks = [num if num in z_ticks else '' for num in arr]

        # plot static figure
        fig = plt.figure(figsize=(12,12))
        ax = sns.heatmap(
            corr2,
            mask=mask,
            xticklabels=z_ticks,
            yticklabels=z_ticks,
            vmin=-1, vmax=1, center=0,
            cmap=cmap,
            square=True,
            cbar_kws=cbar_kws
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='center'
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=45,
            horizontalalignment='center'
        )
        plt.xlabel("element")
        plt.ylabel("element")

        for tick in ax.xaxis.get_major_ticks():
            tick.set_pad(1)
        for tick in ax.yaxis.get_major_ticks():
            tick.set_pad(15)

        plt.savefig(fig_path, bbox_inches='tight', dpi=200)

        # plot interactive figure and save as HTML
        fig_int = px.imshow(corr2, range_color=[-1,1], color_continuous_scale='RdBu')
        fig_int.write_html(html_path, include_plotlyjs='cdn')

        plt.close('all')

