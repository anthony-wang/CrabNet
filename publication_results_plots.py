import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from utils.utils import CONSTANTS
from utils.utils import publication_plot_pred_act, publication_plot_residuals
from use_crabnet import predict_crabnet
from use_densenet import predict_densenet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# %%
plt.rcParams.update({'font.size': 16})

cons = CONSTANTS()
mat_props_units = cons.mp_units_dict
mat_props = cons.mps
mat_props_names = cons.mp_names
pretty_mp_names = cons.mp_names_dict


# %%
def plot_compare_lcs(times,
                     maes,
                     mat_prop,
                     classic_results=None,
                     ax=None):

    mp_sym_dict = cons.mp_sym_dict
    mp_units_dict = cons.mp_units_dict

    fig = None
    if classic_results is not None:
        classic_time = classic_results[0]
        classic_mae = classic_results[1]

    crab_time, dense_time = times
    crab_mae, dense_mae = maes

    x_crab = np.arange(len(crab_mae))
    x_dense = np.arange(len(dense_mae))

    x_crab = np.linspace(0, crab_time, len(crab_mae))
    x_dense = np.linspace(0, dense_time, len(dense_mae))

    # Plot training curve
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x_crab, crab_mae,
            '-', color=cons.crab_red, marker='o', ms=0, alpha=1,
            label='CrabNet')
    ax.plot(x_dense, dense_mae,
            '-', color=cons.dense_blue, marker='s', ms=0, alpha=1,
            label='DenseNet')
    ax.axhline(np.min(dense_mae), color=cons.dense_blue, linestyle='--',
               alpha=1)
    ax.set_xlabel('Training time [s]')

    ax.plot([crab_time, dense_time], [crab_mae.iloc[-5:].mean(),
                                      dense_mae.iloc[-5:].mean()],
            'kX', ms=14, mfc='gold', label='1000 epochs')

    ymax = 1.5*np.mean(dense_mae)

    if classic_results is not None:
        classic_x = classic_time
        classic_y = 1.5*np.mean(dense_mae)

        if classic_time > 1.2 * np.max(crab_time):
            classic_x = np.max(crab_time)
            ax.plot([classic_x*(14/20), classic_x], [classic_mae, classic_mae],
                    'g-', linewidth=5)
            ax.plot(classic_x, classic_mae, '>', mec='green', ms=12,
                    mfc='white', mew=3, label='Best classic')
            ax.text(classic_x, classic_mae, f'({classic_time:0.0f} s)   \n',
                    horizontalalignment='right', verticalalignment='center')
        elif classic_mae > ymax:
            classic_mae = ymax * 0.97
            ax.plot([classic_x, classic_x], [classic_mae*(16.5/20), classic_mae],
                    'g-', linewidth=5)
            ax.plot(classic_x, classic_mae, '^', mec='green', ms=12,
                    mfc='white', mew=3, label='Best classic')
            txt = f'\n\n({classic_mae:0.2f} {mp_units_dict[mat_prop]})     '
            ax.text(classic_x, classic_mae*(16.5/20), txt,
                    horizontalalignment='center', verticalalignment='center')
        else:
            ax.plot(classic_x, classic_mae, 'o', mec='green', ms=12,
                    mfc='white', mew=4, label='Best classic')

    ax.set_ylabel(f'MAE of {mp_sym_dict[mat_prop]} '
                  f'[{mp_units_dict[mat_prop]}]')

    ax.set_ylim(np.min(crab_mae)/1.5, ymax)

    ax.tick_params(left=True, top=True, right=True, direction='in', length=7)
    ax.tick_params(which='minor', left=True, top=True, right=True,
                   direction='in', length=4)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator_x)
    ax.yaxis.set_minor_locator(minor_locator_y)

    # Get all plot labels for legend and label legend
    lines, labels = ax.get_legend_handles_labels()

    ax.legend(lines,
              labels,
              loc='best',
              prop={'size': 12})

    if fig is not None:
        return fig


def multi_plots_lcs(nn_dir, classics_dir):

    files = os.listdir(classics_dir)
    classics_results_csv = classics_dir + [file for file in files
                                           if 'test_scores.csv' in file][0]
    df_classics = pd.read_csv(classics_results_csv)

    files = os.listdir(nn_dir)
    # print(files)
    nn_results_csv = nn_dir + [file for file in files
                               if 'all_results' in file
                               if '.csv' in file][0]
    df_nn = pd.read_csv(nn_results_csv)

    mat_props = df_nn['mat_prop'].unique()
    seeds = df_nn['rng_seed'].unique()
    seed_values = {seed: 0 for seed in seeds}


    df_crabnet = df_nn[df_nn['model_type'] == 'CrabNet']
    for mp in mat_props:
        df_mp = df_crabnet
        mp_bools = df_mp['mat_prop'] == mp
        best_mae = np.min(df_mp[mp_bools]['mae_val'])
        pc_mae = (df_mp[mp_bools]['mae_val'] - best_mae) / best_mae

        imp_col = pd.Series(pc_mae, name='improvement')

        df_mp = pd.concat([df_mp, imp_col], axis=1)
        df_mp = df_mp[df_mp['mat_prop'] == mp].sort_values(by='improvement')
        df_mp_seeds = df_mp['rng_seed']
        for i, seed in enumerate(df_mp_seeds):
            seed_values[seed] += (df_mp.iloc[i]['improvement'])

    ranked_seeds = pd.Series(seed_values).sort_values()
    seed = ranked_seeds.index[0]

    df_nn = df_nn[df_nn['rng_seed'] == seed]


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    mats = ['energy_atom', 'Egap', 'agl_thermal_conductivity_300K',
            'ael_debye_temperature']

    for mp, ax in zip(mats, axes.ravel()):
        run_ids = df_nn[df_nn['mat_prop'] == mp]
        crab_id = run_ids[run_ids['model_type'] == 'CrabNet']['id'].values[0]
        dense_id = run_ids[run_ids['model_type'] == 'DenseNet']['id'].values[0]

        crab_df = pd.read_csv(f'{nn_dir}/{crab_id}/progress.csv')
        dense_df = pd.read_csv(f'{nn_dir}/{dense_id}/progress.csv')
        crab_maes = crab_df['mae_val']
        dense_maes = dense_df['mae_val']

        crab_bools = run_ids['model_type'] == 'CrabNet'
        dense_bools = run_ids['model_type'] == 'DenseNet'
        crab_time = run_ids[crab_bools]['fit_time'].values[0]
        dense_time = run_ids[dense_bools]['fit_time'].values[0]

        df_classic = df_classics[df_classics['mat_prop'] == mp]

        classic_mae = df_classic['mae_test'].values[0]
        classic_time = df_classic['fit_time'].values[0]

        plot_compare_lcs((crab_time, dense_time),
                         (crab_maes, dense_maes),
                         mp,
                         (classic_time, classic_mae),
                         ax=ax)
    plt.subplots_adjust(wspace=0.22)

    out_dir = r'figures/learning_curves/'
    os.makedirs(out_dir, exist_ok=True)

    fig_file = os.path.join(out_dir, f'four_panel_learning_curve.png')
    if fig is not None:
        fig.savefig(fig_file,
                    dpi=300,
                    bbox_inches='tight')


# %%
def plot_dense_crab_preds(mp, ax):
    test_file = f'test_files/{mp}_test.csv'
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    y_act_dense, y_pred_dense = predict_densenet(mp, test_file)
    fig_dense = publication_plot_pred_act(y_act_dense,
                                          y_pred_dense,
                                          mat_prop=mp,
                                          model='DenseNet',
                                          ax=ax[0])

    y_act_crab, y_pred_crab = predict_crabnet(mp, test_file)
    fig_crab = publication_plot_pred_act(y_act_crab,
                                         y_pred_crab,
                                         mat_prop=mp,
                                         model='CrabNet',
                                         ax=ax[1])

    if fig is not None:
        return fig


def multi_plots_preds():
    mat_props = ['energy_atom', 'Egap']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for i, mp in enumerate(mat_props):
        ax = axes[i, :]
        ax = plot_dense_crab_preds(mp, ax)
    plt.subplots_adjust(wspace=0.22)

    out_dir = r'figures/pred_vs_act/'
    os.makedirs(out_dir, exist_ok=True)
    fig_file = os.path.join(out_dir, f'four_panel_pred_vs_act.png')

    if fig is not None:
        fig.savefig(fig_file,
                    dpi=300,
                    bbox_inches='tight')


# %%
def plot_dense_crab_residuals(mp, ax):
    test_file = f'test_files/{mp}_test.csv'
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    y_act_dense, y_pred_dense = predict_densenet(mp, test_file)
    fig_dense = publication_plot_residuals(y_act_dense,
                                           y_pred_dense,
                                           mat_prop=mp,
                                           model='DenseNet',
                                           ax=ax[0])

    y_act_crab, y_pred_crab = predict_crabnet(mp, test_file)
    fig_crab = publication_plot_residuals(y_act_crab,
                                          y_pred_crab,
                                          mat_prop=mp,
                                          model='CrabNet',
                                          ax=ax[1])

    y0_min, y0_max = ax[0].get_ylim()
    y1_min, y1_max = ax[1].get_ylim()

    y_min_min = np.min([y0_min, y1_min])
    y_max_max = np.max([y0_max, y1_max])

    ax[0].set_ylim(y_min_min, y_max_max)
    ax[1].set_ylim(y_min_min, y_max_max)

    if fig is not None:
        return fig


def multi_plots_residuals():
    mat_props = ['energy_atom', 'Egap']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for i, mp in enumerate(mat_props):
        ax = axes[i, :]
        ax = plot_dense_crab_residuals(mp, ax)
    plt.subplots_adjust(wspace=0.22)

    out_dir = r'figures/residuals/'
    os.makedirs(out_dir, exist_ok=True)
    fig_file = os.path.join(out_dir, f'four_panel_residuals.png')

    if fig is not None:
        fig.savefig(fig_file,
                    dpi=300,
                    bbox_inches='tight')


# %%
def get_figures(nn_dir, classics_dir):

    files = os.listdir(classics_dir)
    classics_results_csv = classics_dir + [file for file in files if
                                           'test_scores.csv' in file][0]
    df_classics = pd.read_csv(classics_results_csv)

    files = os.listdir(nn_dir)
    # print(files)
    nn_results_csv = nn_dir + [file for file in files
                               if 'all_results' in file
                               if '.csv' in file][0]
    df_nn = pd.read_csv(nn_results_csv)

    mat_props = df_nn['mat_prop'].unique()
    seeds = df_nn['rng_seed'].unique()
    seed_values = {seed: 0 for seed in seeds}

    df_crabnet = df_nn[df_nn['model_type'] == 'CrabNet']
    for mp in mat_props:
        df_mp = (df_crabnet[df_crabnet['mat_prop'] == mp]
                 .sort_values(by='mae_val'))
        df_mp_seeds = df_mp['rng_seed']
        for i, seed in enumerate(df_mp_seeds):
            seed_values[seed] += i

    ranked_seeds = pd.Series(seed_values).sort_values()
    seed = ranked_seeds.index[0]

    df_nn = df_nn[df_nn['rng_seed'] == seed]
    for mp in mat_props:
        run_ids = df_nn[df_nn['mat_prop'] == mp]
        crab_id = run_ids[run_ids['model_type'] == 'CrabNet']['id'].values[0]
        dense_id = run_ids[run_ids['model_type'] == 'DenseNet']['id'].values[0]

        crab_df = pd.read_csv(f'{nn_dir}/{crab_id}/progress.csv')
        dense_df = pd.read_csv(f'{nn_dir}/{dense_id}/progress.csv')
        crab_maes = crab_df['mae_val']
        dense_maes = dense_df['mae_val']

        crab_bools = run_ids['model_type'] == 'CrabNet'
        dense_bools = run_ids['model_type'] == 'DenseNet'
        crab_time = run_ids[crab_bools]['fit_time'].values[0]
        dense_time = run_ids[dense_bools]['fit_time'].values[0]

        df_classic = df_classics[df_classics['mat_prop'] == mp]

        classic_mae = df_classic['mae_test'].values[0]
        classic_time = df_classic['fit_time'].values[0]

        fig = plot_compare_lcs((crab_time, dense_time),
                               (crab_maes, dense_maes),
                               mp,
                               (classic_time, classic_mae))

        plt.subplots_adjust(wspace=0.22)
        out_dir = r'figures/learning_curves/'
        os.makedirs(out_dir, exist_ok=True)

        fig_file = os.path.join(out_dir, f'{mp}-learning_curve.png')
        fig.savefig(fig_file,
                    dpi=300,
                    bbox_inches='tight')

        out_dir = r'figures/pred_vs_act/'
        os.makedirs(out_dir, exist_ok=True)

        fig_pred_act = plot_dense_crab_preds(mp, ax=None)
        plt.subplots_adjust(wspace=0.22)

        fig_crab_file = os.path.join(out_dir, f'{mp}-pred_vs_act.png')
        fig_pred_act.savefig(fig_crab_file,
                             dpi=300,
                             bbox_inches='tight')

        out_dir = r'figures/residuals/'
        os.makedirs(out_dir, exist_ok=True)

        fig_pred_act = plot_dense_crab_residuals(mp, ax=None)
        plt.subplots_adjust(wspace=0.22)

        fig_crab_file = os.path.join(out_dir, f'{mp}-residuals.png')
        fig_pred_act.savefig(fig_crab_file,
                             dpi=300,
                             bbox_inches='tight')

        plt.close('all')


# %%
def get_test_results(nn_path):
    df_scores = pd.DataFrame()
    for mp in mat_props:
        test_file = f'test_files/{mp}_test.csv'

        y_act_crab, y_pred_crab = predict_crabnet(mp, test_file)
        y_act_dense, y_pred_dense = predict_densenet(mp, test_file)

        df_scores.at[mp, 'r2_crab_test'] = r2_score(y_act_crab, y_pred_crab)
        df_scores.at[mp, 'mae_crab_test'] = mean_absolute_error(y_act_crab,
                                                                y_pred_crab)
        df_scores.at[mp, 'mse_crab_test'] = mean_squared_error(y_act_crab,
                                                               y_pred_crab)
        df_scores.at[mp, 'r2_dense_test'] = r2_score(y_act_dense,
                                                     y_pred_dense)
        df_scores.at[mp, 'mae_dense_test'] = mean_absolute_error(y_act_dense,
                                                                 y_pred_dense)
        df_scores.at[mp, 'mse_dense_test'] = mean_squared_error(y_act_dense,
                                                                y_pred_dense)
    out_path = f'{nn_path}/test_scores.csv'
    df_scores.to_csv(out_path)


# %%
if __name__ == '__main__':
    nn_path = r'data/score_summaries/NN/Run_publication/'
    classics_path = r'data/score_summaries/Classics/Run_publication/'
    get_figures(nn_path, classics_path)
    multi_plots_lcs(nn_path, classics_path)
    multi_plots_preds()
    multi_plots_residuals()
    get_test_results(nn_path)
