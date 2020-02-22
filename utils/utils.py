import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from sklearn.metrics import r2_score

import seaborn as sns

import json

plt.rcParams.update({'font.size': 16})


# %%
fig_dir = r'figures/Classics/'


# %%
class CONSTANTS():
    def __init__(self):
        self.crab_red = '#f2636e'
        self.dense_blue = '#2c2cd5'
        self.colors = list(sns.color_palette("Set1", n_colors=7, desat=0.5))

        self.markers = ['o', 'x', 's', '^', 'D', 'P', '1', '2', '3',
                        '4',  'p', '*', 'h', 'H', '+', 'd',
                        '|', '_']

        self.eps = ['oliynyk',
                    'jarvis',
                    'mat2vec',
                    'onehot',
                    'magpie']

        self.mps = ['ael_shear_modulus_vrh',
                    'energy_atom',
                    'agl_log10_thermal_expansion_300K',
                    'agl_thermal_conductivity_300K',
                    'Egap',
                    'ael_debye_temperature',
                    'ael_bulk_modulus_vrh']

        self.mp_names = ['Log shear modulus',
                         'Ab initio energy per atom',
                         'Log thermal expansion',
                         'Log thermal conductivity',
                         'Band gap',
                         'Debye temperature',
                         'Bulk modulus']

        self.mp_names_dict = dict(zip(self.mps, self.mp_names))

        self.mp_units_dict = {'energy_atom': 'eV/atom',
                              'ael_shear_modulus_vrh': 'GPa',
                              'ael_bulk_modulus_vrh': 'GPa',
                              'ael_debye_temperature': 'K',
                              'Egap': 'eV',
                              'agl_thermal_conductivity_300K': 'W/m*K',
                              'agl_log10_thermal_expansion_300K': '1/K'}

        self.mp_sym_dict = {'energy_atom': '$E_{atom}$',
                            'ael_shear_modulus_vrh': '$G$',
                            'ael_bulk_modulus_vrh': '$B$',
                            'ael_debye_temperature': '$\\theta_D$',
                            'Egap': '$E_g$',
                            'agl_thermal_conductivity_300K': '$\\kappa$',
                            'agl_log10_thermal_expansion_300K': '$\\alpha$'}

        self.classic_models_dict = {'Ridge': 'Ridge',
                                    'SGDRegressor': 'SGD',
                                    'ExtraTreesRegressor': 'ExtraTrees',
                                    'RandomForestRegressor': 'RF',
                                    'AdaBoostRegressor': 'AdaBoost',
                                    'GradientBoostingRegressor': 'GradBoost',
                                    'KNeighborsRegressor': 'kNN',
                                    'SVR': 'SVR',
                                    'lSVR': 'lSVR'}


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()if p.requires_grad)


# %%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# %%
def xstr(s):
    if s is None:
        return ''
    else:
        return f'seed{str(s)}'


def xstrh(s):
    if s is None:
        return ''
    else:
        return xstr(f'{s}-')


# %%
def get_path(score_summary_dir, filename):
    path = os.path.join(score_summary_dir, filename)
    return path


def load_df(path):
    df = pd.read_csv(path)
    return df


# %%
def plot_training_curves(mae_train,
                         mse_train,
                         r2_train,
                         mae_val,
                         mse_val,
                         r2_val,
                         mae_val_max,
                         r2_val_max,
                         model_type,
                         epoch,
                         elem_prop,
                         mat_prop,
                         dataset,
                         optim):

    # Plot training curve
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(0, len(mae_train), 1), mae_train,
             'r--', marker='o', ms=4, alpha=0.5, label='train_mae')
    ax1.plot(np.arange(0, len(mae_val), 1), mae_val,
             'b--', marker='s', ms=4, alpha=0.5, label='val_mae')
    ax1.axhline(mae_val_max, color='b', linestyle='--', alpha=0.3)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel(f'Mean Absolute Error (MAE)')
    ax1.set_ylim(0, 2 * np.mean(mae_val))

    ax1.tick_params(left=True, top=True, direction='in', length=7)
    ax1.tick_params(which='minor', left=True, top=True,
                    direction='in', length=4)

    ax2 = ax1.twinx()
    ax2.set_ylabel('r2')
    ax2.plot(np.arange(0, len(r2_train), 1), r2_train,
             'r-', alpha=0.5, label='train_r2')
    ax2.plot(np.arange(0, len(r2_val), 1), r2_val,
             'b-', alpha=0.5, label='val_r2')
    ax2.axhline(r2_val_max, color='b', alpha=0.3)
    ymin, ymax = (0.4, 1.0)
    ax2.set_ylim(ymin, ymax)
    yticks = np.arange(ymin, ymax + 1e-3, 0.1)
    ax2.set_yticks(yticks)

    ax2.tick_params(right=True, direction='in', length=7)
    ax2.tick_params(which='minor', right=True, direction='in', length=4)

    plt.title(f'net: {model_type}, epoch: {epoch}, '
              f'{elem_prop} with {mat_prop}\n'
              f'dataset: {dataset}\n'
              f'optim: {optim}',
              fontsize=16)

    # Get all plot labels for legend and label legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
               labels + labels2,
               loc='lower left',
               prop={'size': 12})

    return fig


# %%
def plot_pred_act(y_act,
                  y_pred,
                  epoch=None,
                  addn_title_text=None,
                  label=None,
                  outliers=False,
                  **kwargs):
    fig = plt.figure(figsize=(6, 6))

    y_act = np.array(y_act)
    y_pred = np.array(y_pred)

    max_max = np.max([np.max(y_act), np.max(y_pred)])
    min_min = np.min([np.min(y_act), np.min(y_pred)])

    plt.plot(y_act, y_pred, 'o', alpha=0.3, mfc='grey', label=label)
    plt.plot([max_max, min_min], [max_max, min_min], 'k-', alpha=0.7)

    # get and plot outliers
    if outliers:
        outlier_bools = get_outlier_bools(y_act, y_pred, **kwargs)
        plt.plot(y_act[outlier_bools],
                 y_pred[outlier_bools],
                 'x',
                 mfc='red',
                 alpha=0.5,
                 label='outliers')

    plt.xlim(min_min, max_max)
    plt.ylim(min_min, max_max)

    ylocs, ylabels = plt.yticks()

    title_str = f'Performance'
    if epoch or epoch == 0:
        title_str = title_str + f', epoch: {epoch}'
    if addn_title_text:
        title_str = title_str + f', {addn_title_text}'
    plt.title(title_str)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.legend(loc='lower right')

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax = plt.gca()
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    plt.yticks(ylocs)
    plt.xticks(ylocs)

    plt.tick_params(right=True,
                    top=True,
                    direction='in',
                    length=7)
    plt.tick_params(which='minor',
                    right=True,
                    top=True,
                    direction='in',
                    length=4)

    plt.xlim(min_min, max_max)
    plt.ylim(min_min, max_max)

    return fig


# %%
def publication_plot_pred_act(y_act,
                              y_pred,
                              mat_prop,
                              model,
                              ax):

    cons = CONSTANTS()
    mec = cons.crab_red
    mfc = 'silver'
    if model == 'DenseNet':
        mec = cons.dense_blue
        mfc = 'silver'

    mp_sym_dict = cons.mp_sym_dict
    mp_units_dict = cons.mp_units_dict

    y_act = np.array(y_act)
    y_pred = np.array(y_pred)

    ymin = np.min([y_act]) * 0.9
    ymax = np.max([y_act]) / 0.9
    xmin = ymin
    xmax = ymax

    ax.plot(y_act, y_pred, 'o', ms=10, mec=mec, mfc=mfc,
            alpha=0.35, label=model)
    ax.plot([xmin, xmax], [ymin, ymax], 'k--', alpha=0.7, label='Ideal')

    r2 = r2_score(y_act, y_pred)
    ax.text(xmin + np.abs(xmin-xmax)*0.04,
            ymax - np.abs(ymin-ymax)*0.05,
            f'$r^2$ = {r2:0.3f}',
            horizontalalignment='left',
            verticalalignment='top')

    ax.set_xlabel(f'Actual {mp_sym_dict[mat_prop]} '
                  f'[{mp_units_dict[mat_prop]}]')
    ax.set_ylabel(f'Predicted {mp_sym_dict[mat_prop]} '
                  f'[{mp_units_dict[mat_prop]}]')
    ax.legend(loc='lower right')

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax


# %%

def publication_plot_residuals(y_act,
                               y_pred,
                               mat_prop,
                               model,
                               ax):

    cons = CONSTANTS()
    mec = cons.crab_red
    mfc = 'silver'
    if model == 'DenseNet':
        mec = cons.dense_blue
        mfc = 'silver'

    mp_sym_dict = cons.mp_sym_dict
    mp_units_dict = cons.mp_units_dict

    y_act = np.array(y_act)
    y_pred = np.array(y_pred)

    xmin = np.min([y_act]) * 0.9
    xmax = np.max([y_act]) / 0.9

    y_err = y_pred - y_act
    ymin = np.min([y_err]) * 0.9
    ymax = np.max([y_err]) / 0.9

    ax.plot(y_act, y_err, 'o', ms=10, mec=mec, mfc=mfc,
            alpha=0.35, label=model)
    ax.plot([xmin, xmax], [0, 0], 'k--', alpha=0.7)

    ax.set_xlabel(f'Actual {mp_sym_dict[mat_prop]} '
                  f'[{mp_units_dict[mat_prop]}]')
    ax.set_ylabel(f'Residual {mp_sym_dict[mat_prop]} '
                  f'[{mp_units_dict[mat_prop]}]')
    ax.legend(loc='lower right')

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax


# %%
def get_outlier_bools(y_act, y_pred, threshold=0.4):
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    threshold = float(threshold)

    bool_outliers1 = np.abs(np.abs(y_act - y_pred) / y_act) > threshold
    bool_outliers2 = np.abs(np.abs(y_act - y_pred) / y_act) > threshold
    bool_outliers = bool_outliers1 + bool_outliers2

    bool_outliers = y_pred > y_act*(1+threshold) + y_pred < y_act*(1-threshold)

    print(f'number of outliers: {np.sum(bool_outliers)}')

    return bool_outliers


# %%
def plot_best_gs_models_ep(summaries_list,
                           score_metric,
                           mat_props,
                           elem_props,
                           score_summary_dir,
                           fig_dir):
    """
    Plot the best gridsearch test scores for each property for each
    featurization scheme, grouped by model.
    Parameters
    ----------
    summaries_list : TYPE
        DESCRIPTION.
    score_metric : TYPE
        DESCRIPTION.
    mat_props : TYPE
        DESCRIPTION.
    elem_props : TYPE
        DESCRIPTION.
    Returns
    -------
    Saves plots in the plot directory.
    """
    cons = CONSTANTS()
    colors = cons.colors
    markers = cons.markers
    classic_models_dict = cons.classic_models_dict
    mp_names_dict = cons.mp_names_dict

    for mp in mat_props:
        files = [file for file in summaries_list if mp in file]
        df = pd.DataFrame()

        for file in files:
            path = get_path(score_summary_dir, file)
            df_file = pd.read_csv(path)
            df_file['elem_prop'] = file.split('_')[1]
            df = pd.concat([df, df_file], axis=0)

        fig, ax = plt.subplots(figsize=(6, 6))
        for i, ep in enumerate(elem_props):
            df_temp = df.loc[df['elem_prop'] == ep]
            plt.plot(df_temp['estimator'],
                     df_temp['mean_test_r2'],
                     color=colors[i],
                     marker=markers[i],
                     alpha=0.8,
                     label=ep)

        plt.xticks(np.arange(len(df_temp['estimator'])),
                   [classic_models_dict.get(k) for k in df_temp['estimator']],
                   rotation=45)

        plt.tick_params(left=True, top=True, right=True,
                        direction='in', length=7)
        plt.tick_params(which='minor', left=True, top=True, right=True,
                        direction='in', length=4)

        minor_locator_y = AutoMinorLocator(2)
        ax.yaxis.set_minor_locator(minor_locator_y)

        plt.legend(title='Featurization scheme', prop={'size': 14})
        plt.ylabel(score_metric)
        plt.xlabel('model')
        plt.ylim(-0.4, 1)
        plt.title(mp_names_dict[mp], fontsize=18)

        outpath = os.path.join(fig_dir, f'{mp}.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print('done saving', outpath)
        plt.close('all')


# %%
if __name__ == '__main__':
    os.makedirs(fig_dir, exist_ok=True)

    # score_metric = 'mean_test_r2'
    # cons = CONSTANTS()
    # mat_props = cons.mps
    # elem_props = cons.eps
    # plot_best_gs_models_ep(summaries_list,
    #                        score_metric,
    #                        mat_props,
    #                        elem_props,
    #                        fig_dir)
