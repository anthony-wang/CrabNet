import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.utils import CONSTANTS


# %%
cons = CONSTANTS()

for benchmark in [True, False]:
    df = pd.read_csv('metrics/all_metrics.csv')
    drop = ['n_total', 'n_val',
            'prop_train', 'prop_val', 'prop_test']
    df = df.drop(columns=drop, axis=0)
    df.index = df['property'].values

    if benchmark:
        convert = cons.benchmark_names_dict
        data_dir = 'data/benchmark_data/'
        properties = os.listdir(data_dir)

    else:
        convert = cons.matbench_names_dict
        data_dir = 'data/matbench/'
        drop = ['expt_is_metal', 'glass', 'mp_is_metal']
        properties = [p[:-4] for p in os.listdir(data_dir) if p[:-4] not in drop]

    data_values = df.loc[properties].values


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
        pal = sns.diverging_palette(240, 10,
                                    n=n_colors * 1.0)
        diff = value - min_val
        idx = int((n_colors - 1) * (diff) / (max_val - min_val))
        color = pal[idx]

        if value < 0:
            color = '#fab098'

        return color


    # %%

    if benchmark:
        fig, ax = plt.subplots(figsize=(13, 9))
    else:
        fig, ax = plt.subplots(figsize=(13, 5))

    cell_h = 1
    cell_w = 1
    cell_width = 7
    left_bound = 14
    n_columns = 6

    for i, array in enumerate(data_values):

        denominator = 75  # add buffer to color scale bounds "max + std/demoniator"
        r2_max = np.nanmax(array[1:n_columns]) + array[n_columns+4]/denominator
        r2_min = np.nanmin(array[1:n_columns]) - array[n_columns+4]/denominator

        if i == 0:
            for j in range(1, n_columns):
                rect = Rectangle([j*cell_width, len(data_values)],
                                 cell_width * cell_w,
                                 1 * cell_h,
                                 facecolor='w',
                                 edgecolor='k')
                ax.add_patch(rect)
                rect = Rectangle([j*cell_width, len(data_values)],
                                 cell_width * cell_w,
                                 1 * cell_h,
                                 facecolor='w',
                                 edgecolor='k')
                ax.add_patch(rect)
                rect = Rectangle([j*cell_width, len(data_values)],
                                 cell_width * cell_w,
                                 1 * cell_h,
                                 facecolor='w',
                                 edgecolor='k')
                ax.add_patch(rect)

        for j, value in enumerate(array):
            # print(value)
            # j = j
            if j == 0:
                rect = Rectangle([j-left_bound, i],
                                  (left_bound+cell_width) * cell_w,
                                  1 * cell_h,
                                  facecolor='w', edgecolor='k')
                ax.add_patch(rect)

            if j >= 1 and j < n_columns:
                # print(array[0], j)
                value = float(value)
                if np.isnan(value) or np.isnan(r2_min) or np.isnan(r2_max):
                    print('nan values found')
                    rect = Rectangle([j*cell_width, len(data_values) - i - 1],
                                     cell_width * cell_w,
                                     1 * cell_h,
                                     facecolor='gray',
                                     edgecolor='k')
                else:
                    if j in [j * cell_w for j in range(len(array))]:

                        # if j == n_columns-1:
                        if value > r2_max or value < r2_min:
                            rect = Rectangle([j*cell_width,
                                              len(data_values) - i - 1],
                                             cell_width * cell_w,
                                             1 * cell_h,
                                             facecolor='gray',
                                             edgecolor='k')
                        else:
                            rect = Rectangle([j*cell_width,
                                              len(data_values) - i - 1],
                                         cell_width * cell_w,
                                         1 * cell_h,
                                         facecolor=color_scale_mae(value,
                                                                   r2_min,
                                                                   r2_max),
                                         edgecolor='k')
                ax.add_patch(rect)

            if j == 0:
                eformat = False
                if value == 'aflow__agl_thermal_expansion_300K':
                    eformat = True
                plt.text(j - left_bound + (0.5 * cell_w),
                         len(data_values) - (i + (0.5 * cell_h)),
                         f'{convert[value]}',
                         verticalalignment='center',
                         horizontalalignment='left')

            elif j >= 1 and j < n_columns:
                if eformat:
                    plt.text(j*cell_width + (1 * cell_width/2),
                             len(data_values) - (i + (0.5 * cell_h)),
                             f'{value:0.2e}',
                             verticalalignment='center',
                             horizontalalignment='center')
                else:
                    plt.text(j*cell_width + (1 * cell_width/2),
                             len(data_values) - (i + (0.5 * cell_h)),
                             f'{value:0.3f}',
                             verticalalignment='center',
                             horizontalalignment='center')


    plt.xlim(0-left_bound, cell_width*n_columns)
    plt.ylim(0, len(data_values)*cell_h+1)

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)

    if benchmark:
        plt.text(-left_bound + (cell_width + left_bound)/2,
                 len(data_values)*cell_h+0.5, 'Extended Properties',
        verticalalignment='center',
        horizontalalignment='center')
    else:
        plt.text(-left_bound + (cell_width + left_bound)/2,
                 len(data_values)*cell_h+0.5, 'MatBench Properties',
        verticalalignment='center',
        horizontalalignment='center')

    plt.text(1*cell_width + (1 * cell_width/2),
             len(data_values)*cell_h+0.5, 'Roost',
    verticalalignment='center',
    horizontalalignment='center')

    plt.text(2*cell_width + (1 * cell_width/2),
             len(data_values)*cell_h+0.5, 'CrabNet',
    verticalalignment='center',
    horizontalalignment='center')

    plt.text(3*cell_width + (1 * cell_width/2),
             len(data_values)*cell_h+0.5, 'HotCrab',
    verticalalignment='center',
    horizontalalignment='center')

    plt.text(4*cell_width + (1 * cell_width/2),
             len(data_values)*cell_h+0.5, 'ElemNet',
    verticalalignment='center',
    horizontalalignment='center')

    plt.text(5*cell_width + (1 * cell_width/2),
             len(data_values)*cell_h+0.5, 'RF',
    verticalalignment='center',
    horizontalalignment='center')


    outpath = os.path.join('figures/')

    if benchmark:
        save_name = os.path.join(outpath, 'Table2_benchmark_metrics.png')
    else:
        save_name = os.path.join(outpath, 'Table2_matbench_metrics.png')

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

