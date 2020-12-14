import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, roc_auc_score, r2_score
import numpy as np
import matplotlib.pyplot as plt

normalize = True
normalize = False


# %%
benchmark_statistics = pd.DataFrame()
data_dir = '../data/benchmark_data/'
for data in os.listdir(data_dir):
    df = pd.read_csv(f'{data_dir}{data}/train.csv')
    benchmark_statistics.loc[data, 'mean_train'] = df['target'].mean()
    benchmark_statistics.loc[data, 'std_train'] = df['target'].std()
    benchmark_statistics.loc[data, 'n_train'] = len(df['target'])


matbench_statistics = pd.DataFrame()
data_dir = '../data/matbench/'
for data in os.listdir(data_dir):
    df = pd.read_csv(f'{data_dir}{data}')
    matbench_statistics.loc[data[:-4], 'mean_train'] = df['target'].mean()
    matbench_statistics.loc[data[:-4], 'std_train'] = df['target'].std()
    matbench_statistics.loc[data[:-4], 'n_train'] = int(0.72*len(df['target']))

df_train_stats = pd.concat([benchmark_statistics, matbench_statistics])


# %%
roost_dir = '../publication_predictions/roost_benchmark__predictions'
roost = os.listdir(roost_dir)
crabnet_dir = '../publication_predictions/mat2vec_benchmark__predictions'
crabnet = os.listdir(crabnet_dir)
hotcrab_dir = '../publication_predictions/onehot_benchmark__predictions'
hotcrab = os.listdir(hotcrab_dir)
elemnet_dir = '../publication_predictions/elemnet_benchmark__predictions'
elemnet = os.listdir(elemnet_dir)
rf_dir = '../publication_predictions/rf_benchmark__predictions'
rf = os.listdir(rf_dir)

df_master = pd.DataFrame()
df_r2 = pd.DataFrame()


for mat_pred in roost:
    mat_prop = mat_pred.split('test_results_')[-1].split('_model')[0]
    df_roost = pd.read_csv(f'{roost_dir}/{mat_pred}')
    act, pred = df_roost['target'], df_roost['pred-0']
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    df_master.loc[mat_prop, 'roost'] = mean_absolute_error(act, pred)
    df_r2.loc[mat_prop, 'roost'] = r2_score(act, pred)


for mat_pred in crabnet:
    if '_test_output' in mat_pred:
        mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
        df_crabnet = pd.read_csv(f'{crabnet_dir}/{mat_pred}')
        act, pred = df_crabnet['target'], df_crabnet['pred-0']
        if normalize:
            act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
        df_master.loc[mat_prop, 'crabnet'] = mean_absolute_error(act, pred)
        df_r2.loc[mat_prop, 'crabnet'] = r2_score(act, pred)


for mat_pred in hotcrab:
    if '_test_output' in mat_pred:
        mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
        df_hotcrab = pd.read_csv(f'{hotcrab_dir}/{mat_pred}')
        act, pred = df_hotcrab['target'], df_hotcrab['pred-0']
        if normalize:
            act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
        df_master.loc[mat_prop, 'hotcrab'] = mean_absolute_error(act, pred)
        df_r2.loc[mat_prop, 'hotcrab'] = r2_score(act, pred)


for mat_pred in elemnet:
    mat_prop = mat_pred.split('_test')[0]
    df_elemnet = pd.read_csv(f'{elemnet_dir}/{mat_pred}')
    act, pred = df_elemnet['actual'], df_elemnet['predicted']
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    df_master.loc[mat_prop, 'elemnet'] = mean_absolute_error(act, pred)
    df_r2.loc[mat_prop, 'elemnet'] = r2_score(act, pred)


for mat_pred in rf:
    mat_prop = mat_pred.split('_test')[0]
    df_rf = pd.read_csv(f'{rf_dir}/{mat_pred}')
    act, pred = df_rf['target'], df_rf['pred']
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    df_master.loc[mat_prop, 'rf'] = mean_absolute_error(act, pred)
    df_r2.loc[mat_prop, 'rf'] = r2_score(act, pred)


for mat_pred in crabnet:
    if '_test_output' in mat_pred:
        mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
        df_crabnet = pd.read_csv(f'{crabnet_dir}/{mat_pred}')
        act, pred = df_crabnet['target'], df_crabnet['pred-0']
        if normalize:
            act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
        df_master.loc[mat_prop, 'mean_test'] = act.mean()
        df_master.loc[mat_prop, 'std_test'] = act.std()
        df_master.loc[mat_prop, 'n_test'] = len(act)
        df_r2.loc[mat_prop, 'n_test'] = len(act)


df_r2_benchmark = df_r2.copy()
df_benchmark = df_master.copy()
df_normalized_benchmark = df_benchmark / df_benchmark[['std_test']].values


# %%
df_metrics = pd.DataFrame()
classification_list = []

roost_dir = '../publication_predictions/roost_matbench__predictions'
roost = os.listdir(roost_dir)
crabnet_dir = '../publication_predictions/mat2vec_matbench__predictions'
crabnet = os.listdir(crabnet_dir)
hotcrab_dir = '../publication_predictions/onehot_matbench__predictions'
hotcrab = os.listdir(crabnet_dir)
elemnet_dir = '../publication_predictions/elemnet_matbench__predictions'
elemnet = os.listdir(elemnet_dir)
rf_dir = '../publication_predictions/rf_matbench__predictions'
rf = os.listdir(rf_dir)


roost_scores = {prop: [] for prop in os.listdir('../data/matbench_cv')}
for mat_pred in roost:
    if '_val' in mat_pred:
        continue
    mat_prop = mat_pred.split('test_results_')[-1].split('_model')[0]
    df_roost = pd.read_csv(f'{roost_dir}/{mat_pred}')
    act, pred = df_roost['target'].values, df_roost['pred-0'].values
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    if mat_prop[:-1] in classification_list:
        act[act != 0] = 1
        roost_scores[mat_prop[:-1]].append(roc_auc_score(act, pred))
    else:
        roost_scores[mat_prop[:-1]].append(mean_absolute_error(act, pred))


crabnet_scores = {prop: [] for prop in os.listdir('../data/matbench_cv')}
for mat_pred in crabnet:
    if '_val' in mat_pred:
        continue
    mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
    mat_prop += mat_pred.rstrip('.csv')[-1]
    df_crabnet = pd.read_csv(f'{crabnet_dir}/{mat_pred}')
    act, pred = df_crabnet['target'].values, df_crabnet['pred-0'].values
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    if mat_prop[:-1] in classification_list:
        act[act != 0] = 1
        crabnet_scores[mat_prop[:-1]].append(roc_auc_score(act, pred))
    else:
        crabnet_scores[mat_prop[:-1]].append(mean_absolute_error(act, pred))


hotcrab_scores = {prop: [] for prop in os.listdir('../data/matbench_cv')}
for mat_pred in hotcrab:
    if '_val' in mat_pred:
        continue
    mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
    mat_prop += mat_pred.rstrip('.csv')[-1]
    df_crabnet = pd.read_csv(f'{hotcrab_dir}/{mat_pred}')
    act, pred = df_crabnet['target'].values, df_crabnet['pred-0'].values
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    if mat_prop[:-1] in classification_list:
        act[act != 0] = 1
        hotcrab_scores[mat_prop[:-1]].append(roc_auc_score(act, pred))
    else:
        hotcrab_scores[mat_prop[:-1]].append(mean_absolute_error(act, pred))


elemnet_scores = {prop: [] for prop in os.listdir('../data/matbench_cv')}
for mat_pred in elemnet:
    if '_val' in mat_pred:
        continue
    mat_prop = mat_pred.split('_test')[0]
    df_elemnet = pd.read_csv(f'{elemnet_dir}/{mat_pred}')
    act, pred = df_elemnet['actual'].values, df_elemnet['predicted'].values
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    if mat_prop in classification_list:
        act[act != 0] = 1
        elemnet_scores[mat_prop].append(roc_auc_score(act, pred))
    else:
        elemnet_scores[mat_prop].append(mean_absolute_error(act, pred))


rf_scores = {prop: [] for prop in os.listdir('../data/matbench_cv')}
for mat_pred in rf:
    mat_prop = mat_pred.split('_test')[0]
    df_rf = pd.read_csv(f'{rf_dir}/{mat_pred}')
    act, pred = df_rf['target'].values, df_rf['pred'].values
    if mat_prop in classification_list:
        act[act != 0] = 1
        rf_scores[mat_prop].append(roc_auc_score(act, pred))
    else:
        rf_scores[mat_prop].append(mean_absolute_error(act, pred))


df_master = pd.DataFrame()
for key, value in roost_scores.items():
    score = np.nan
    if len(value) != 0:
        score = sum(value)/len(value)
    df_master.loc[key, 'roost'] = score

for key, value in crabnet_scores.items():
    score = np.nan
    if len(value) != 0:
        score = sum(value)/len(value)
    df_master.loc[key, 'crabnet'] = score

for key, value in hotcrab_scores.items():
    score = np.nan
    if len(value) != 0:
        score = sum(value)/len(value)
    df_master.loc[key, 'hotcrab'] = score

for key, value in elemnet_scores.items():
    score = np.nan
    if len(value) != 0:
        score = sum(value)/len(value)
    df_master.loc[key, 'elemnet'] = score

for key, value in rf_scores.items():
    score = np.nan
    if len(value) != 0:
        score = sum(value)/len(value)
    df_master.loc[key, 'rf'] = score


crabnet = crabnet[::5]
for mat_pred in crabnet:
    if '_val' in mat_pred:
        continue
    mat_prop = mat_pred.split('test_results_')[-1].split('_test_output')[0]
    mat_prop += mat_pred.rstrip('.csv')[-1]
    df_crabnet = pd.read_csv(f'{crabnet_dir}/{mat_pred}')
    act, pred = df_crabnet['target'].values, df_crabnet['pred-0'].values
    if normalize:
        act, pred = (act-act.mean())/act.std(), (pred-act.mean())/act.std()
    df_master.loc[mat_prop[:-1], 'mean_test'] = act.mean()
    df_master.loc[mat_prop[:-1], 'std_test'] = act.std()
    df_master.loc[mat_prop[:-1], 'n_test'] = len(act)

df_matbench = df_master.copy()


# %%
df_all = pd.concat([df_benchmark, df_matbench])
df_all = pd.concat([df_all, df_train_stats], axis=1)

plt.plot(df_all['n_test'], df_all['elemnet'], 'k*', label='ElemNet')
plt.plot(df_all['n_test'], df_all['roost'], 'o', label='Roost')
plt.plot(df_all['n_test'], df_all['crabnet'], 'rx', label='CrabNet')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('mae')
plt.xlabel('n_test')
plt.legend(loc='lower left')


# %%
if normalize:
    df_all.to_csv('all_metrics_normalized.csv', index_label='property')
else:
    df_all.to_csv('all_metrics.csv', index_label='property')
