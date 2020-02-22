import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from utils.utils import CONSTANTS

# %%
cons = CONSTANTS()

data_dir = 'data/score_summaries/'
file_name = 'CrabNet_vs_DenseNet_vs_Classics.csv'

df_data = pd.read_csv(f'{data_dir}/{file_name}')
df_data = df_data.iloc[::-1]

data_values = df_data.values

mae_idx = [7]
mae_max = data_values[:-1, mae_idx].astype(float).max()
mae_min = data_values[:-1, mae_idx].astype(float).min()

r2_idx = [1, 2, 3]
r2_cols = slice(1, 4)
r2_max = data_values[:-1, r2_cols].astype(float).max()
r2_min = data_values[:-1, r2_cols].astype(float).min()


# %%
def color_scale_r2(value, min_val, max_val):
    n_colors = 100
    pal = sns.light_palette((210, 80, 50),
                            input="husl",
                            n_colors=n_colors * 1.0)
    diff = value - min_val
    idx = int((n_colors - 1) * (diff) / (max_val - min_val))
    color = pal[idx]
    return color


def color_scale_mae(value, min_val, max_val):
    n_colors = 100
    pal = sns.light_palette((210, 80, 50),
                            input="husl",
                            n_colors=n_colors * 1.0)
    diff = value - min_val
    idx = int((n_colors - 1) * (diff) / (max_val - min_val))
    color = pal[idx]

    if value < 0:
        color = '#fab098'

    return color


# %%

fig, ax = plt.subplots(figsize=(12.5, 5))

cell_h = 1
cell_w = 1

for i, array in enumerate(data_values):
    # i *= cell_h
    # print(array)
    for j, value in enumerate(array):
        if i == len(data_values) - 1:
            # print(value)
            plt.text(j + (0.5 * cell_w),
                     i + (0.5 * cell_h),
                     f'{value}',
                     verticalalignment='center',
                     horizontalalignment='center')
            rect = Rectangle([j, i],
                             1 * cell_w,
                             1 * cell_h,
                             facecolor='15',
                             edgecolor='k')
            ax.add_patch(rect)
        else:
            if j > 0:
                value = float(value)
            if j in [j * cell_w for j in r2_idx]:
                rect = Rectangle([j, i],
                                 1 * cell_w,
                                 1 * cell_h,
                                 facecolor=color_scale_r2(value, r2_min, r2_max),
                                 edgecolor='k')
            elif j in [j * cell_w for j in mae_idx]:
                rect = Rectangle([j, i],
                                 1 * cell_w,
                                 1 * cell_h,
                                 facecolor=color_scale_mae(value, mae_min, mae_max),
                                 edgecolor='k')
            else:
                rect = Rectangle([j, i],
                                 1 * cell_w,
                                 1 * cell_h,
                                 facecolor='w', edgecolor='k')
            ax.add_patch(rect)
            if j == 0:
                if value == 'property':
                    plt.text(j + (0.5 * cell_w),
                             i + (0.5 * cell_h),
                             f'{value}',
                             verticalalignment='center',
                             horizontalalignment='center')
                else:
                    plt.text(j + (0.5 * cell_w),
                             i + (0.5 * cell_h),
                             f'{cons.mp_sym_dict[value]}',
                             verticalalignment='center',
                             horizontalalignment='center')
            else:
                plt.text(j + (0.5 * cell_w),
                         i + (0.5 * cell_h),
                         f'{value:0.3f}',
                         verticalalignment='center',
                         horizontalalignment='center')

i = data_values.shape[0] + 1
rect = Rectangle([0, data_values.shape[0]],
                 data_values.shape[-1],
                 1,
                 facecolor='15',
                 edgecolor='k')
ax.add_patch(rect)
rect = Rectangle([0, data_values.shape[0]],
                 1, 1,
                 facecolor='15',
                 edgecolor='k')
ax.add_patch(rect)
rect = Rectangle([1, data_values.shape[0]],
                 3, 1,
                 facecolor='15',
                 edgecolor='k')
ax.add_patch(rect)
ax.add_patch(rect)

i = i - 1

plt.text(2 + (0.5 * cell_w), i + (0.5 * cell_h), f'$r^2$ scores on test set',
verticalalignment='center',
horizontalalignment='center')

plt.text(5.5 + (0.5 * cell_w), i + (0.5 * cell_h), f'MAE scores on test set',
verticalalignment='center',
horizontalalignment='center')

plt.xlim(0, 8*cell_w)
plt.ylim(0, 9*cell_h)

plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)


out_path = 'figures/CrabNet_vs_DenseNet_vs_Classics.png'
plt.savefig(out_path,
            dpi=300,
            bbox_inches='tight')
plt.close('all')