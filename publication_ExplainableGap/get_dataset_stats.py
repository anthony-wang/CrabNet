import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.utils import CONSTANTS
from utils.composition import _element_composition

from collections import Counter

from matplotlib import pyplot as plt

from scipy.stats import entropy as Entropy


# %%
benchmark_data = os.listdir('data/benchmark_data')
matbench_data = os.listdir('data/matbench_cv')

fig_path = 'figures/element_prevalence/'

cons = CONSTANTS()
symbol_idx_dict = cons.symbol_idx_dict



# %%
def calc_diversity(df):
    count_col = df['count']
    diversity = Entropy(count_col, base=2) / (np.log2(len(count_col)))
    return diversity


# %%
cols = ['mat_prop', 'n_elements', 'samples_total',
        'samples_train', 'samples_val', 'samples_test',
        'prop_train', 'prop_val', 'prop_test',
        'train_diversity', 'val_diversity', 'test_diversity']
df_benchmark = pd.DataFrame(columns=cols)

for prop in tqdm(benchmark_data, desc="Processing benchmark_data"):
    df_train = pd.read_csv(f'data/benchmark_data/{prop}/train.csv',
                           keep_default_na=False, na_values=[''])
    df_val = pd.read_csv(f'data/benchmark_data/{prop}/val.csv',
                         keep_default_na=False, na_values=[''])
    df_test = pd.read_csv(f'data/benchmark_data/{prop}/test.csv',
                          keep_default_na=False, na_values=[''])

    train_dicts = df_train['formula'].apply(_element_composition)
    val_dicts = df_val['formula'].apply(_element_composition)
    test_dicts = df_test['formula'].apply(_element_composition)

    train_list = [item for row in train_dicts for item in row.keys()]
    val_list = [item for row in val_dicts for item in row.keys()]
    test_list = [item for row in test_dicts for item in row.keys()]

    train_counter = Counter(train_list)
    val_counter = Counter(val_list)
    test_counter = Counter(test_list)
    trainc_df = pd.DataFrame.from_dict(train_counter, orient='index', columns=['count'])
    valc_df = pd.DataFrame.from_dict(val_counter, orient='index', columns=['count'])
    testc_df = pd.DataFrame.from_dict(test_counter, orient='index', columns=['count'])

    train_diversity = calc_diversity(trainc_df)
    val_diversity = calc_diversity(testc_df)
    test_diversity = calc_diversity(valc_df)

    trainc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in trainc_df.index]
    valc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in valc_df.index]
    testc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in testc_df.index]

    n_train = train_dicts.apply(len).max()
    n_val = val_dicts.apply(len).max()
    n_test = test_dicts.apply(len).max()
    n_elements = max([n_train, n_val, n_test])
    total_samples = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
    df_row = {
        'mat_prop': prop,
        'n_elements': n_elements,
        'samples_total': total_samples,
        'samples_train': df_train.shape[0],
        'samples_val': df_val.shape[0],
        'samples_test': df_test.shape[0],
        'prop_train': df_train.shape[0] / total_samples,
        'prop_val': df_val.shape[0] / total_samples,
        'prop_test': df_test.shape[0] / total_samples,
        'train_diversity': train_diversity,
        'val_diversity': val_diversity,
        'test_diversity': test_diversity,
        }
    df_benchmark = df_benchmark.append(df_row, ignore_index=True)

    most_common = train_counter.most_common()
    df_most_common = pd.DataFrame(most_common, columns=['element', 'counts'])
    df_most_common['atomic_number'] = [symbol_idx_dict[atom] for atom in df_most_common['element']]

    elements = [f'{z}: {el}' for el, z in zip(df_most_common['element'], df_most_common['atomic_number'])]
    plt.figure(figsize=(20,5))
    plt.bar(elements, df_most_common['counts'])
    plt.xticks(fontsize=10, rotation=45, rotation_mode='anchor', horizontalalignment='right')
    plt.title(prop + ', train')
    plt.savefig(f'{fig_path}/{prop}_byCounts.png', bbox_inches='tight', dpi=200)

    plt.figure(figsize=(20,5))
    df_most_common = df_most_common.sort_values(by='atomic_number')
    elements = [f'{z}: {el}' for el, z in zip(df_most_common['element'], df_most_common['atomic_number'])]
    plt.bar(elements, df_most_common['counts'])
    plt.xticks(fontsize=10, rotation=45, rotation_mode='anchor', horizontalalignment='right')
    plt.title(prop + ', train')
    plt.savefig(f'{fig_path}/{prop}_byZ.png', bbox_inches='tight', dpi=200)


# plot Shannon diversity for benchmark datasets
fig = plt.figure(figsize=(20,8))
y_pos = np.arange(len(benchmark_data))
plt.barh(y_pos, df_benchmark['train_diversity'], tick_label=benchmark_data)
ax_list = fig.axes
ax_list[0].invert_yaxis()
plt.title('Shannon diversity index for benchmark datasets')
plt.savefig(f'{fig_path}/shannon_diversity_benchmark_data.png', bbox_inches='tight', dpi=200)

plt.close('all')


# %%
cols = ['mat_prop', 'cv', 'n_elements', 'samples_total',
        'samples_train', 'samples_val', 'samples_test',
        'prop_train', 'prop_val', 'prop_test',
        'train_diversity', 'val_diversity', 'test_diversity']
df_matbench_cv = pd.DataFrame(columns=cols)

for prop in tqdm(matbench_data, desc="Processing matbench_data"):
    for cv in tqdm(range(5), desc="CV:"):
        df_train = pd.read_csv(f'data/matbench_cv/{prop}/train{cv}.csv',
                               keep_default_na=False, na_values=[''])
        df_val = pd.read_csv(f'data/matbench_cv/{prop}/val{cv}.csv',
                             keep_default_na=False, na_values=[''])
        df_test = pd.read_csv(f'data/matbench_cv/{prop}/test{cv}.csv',
                              keep_default_na=False, na_values=[''])

        train_dicts = df_train['formula'].apply(_element_composition)
        val_dicts = df_val['formula'].apply(_element_composition)
        test_dicts = df_test['formula'].apply(_element_composition)

        train_list = [item for row in train_dicts for item in row.keys()]
        val_list = [item for row in val_dicts for item in row.keys()]
        test_list = [item for row in test_dicts for item in row.keys()]

        train_counter = Counter(train_list)
        val_counter = Counter(val_list)
        test_counter = Counter(test_list)
        trainc_df = pd.DataFrame.from_dict(train_counter, orient='index', columns=['count'])
        valc_df = pd.DataFrame.from_dict(val_counter, orient='index', columns=['count'])
        testc_df = pd.DataFrame.from_dict(test_counter, orient='index', columns=['count'])

        train_diversity = calc_diversity(trainc_df)
        val_diversity = calc_diversity(testc_df)
        test_diversity = calc_diversity(valc_df)

        trainc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in trainc_df.index]
        valc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in valc_df.index]
        testc_df['atomic_number'] = [symbol_idx_dict[atom] for atom in testc_df.index]

        n_train = train_dicts.apply(len).max()
        n_val = val_dicts.apply(len).max()
        n_test = test_dicts.apply(len).max()
        n_elements = max([n_train, n_val, n_test])
        total_samples = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
        df_row = {
            'mat_prop': prop,
            'cv': cv,
            'n_elements': n_elements,
            'samples_total': total_samples,
            'samples_train': df_train.shape[0],
            'samples_val': df_val.shape[0],
            'samples_test': df_test.shape[0],
            'prop_train': df_train.shape[0] / total_samples,
            'prop_val': df_val.shape[0] / total_samples,
            'prop_test': df_test.shape[0] / total_samples,
            'train_diversity': train_diversity,
            'val_diversity': val_diversity,
            'test_diversity': test_diversity,
            }
        df_matbench_cv = df_matbench_cv.append(df_row, ignore_index=True)

        most_common = train_counter.most_common()
        df_most_common = pd.DataFrame(most_common, columns=['element', 'counts'])
        df_most_common['atomic_number'] = [symbol_idx_dict[atom] for atom in df_most_common['element']]

        elements = [f'{z}: {el}' for el, z in zip(df_most_common['element'], df_most_common['atomic_number'])]
        plt.figure(figsize=(20,5))
        plt.bar(elements, df_most_common['counts'])
        plt.xticks(fontsize=10, rotation=45, rotation_mode='anchor', horizontalalignment='right')
        plt.title(f'{prop} cv{cv}, train')
        plt.savefig(f'{fig_path}/{prop}{cv}_byCounts.png', bbox_inches='tight', dpi=200)

        plt.figure(figsize=(20,5))
        df_most_common = df_most_common.sort_values(by='atomic_number')
        elements = [f'{z}: {el}' for el, z in zip(df_most_common['element'], df_most_common['atomic_number'])]
        plt.bar(elements, df_most_common['counts'])
        plt.xticks(fontsize=10, rotation=45, rotation_mode='anchor', horizontalalignment='right')
        plt.title(f'{prop} cv{cv}, train')
        plt.savefig(f'{fig_path}/{prop}{cv}_byZ.png', bbox_inches='tight', dpi=200)

    plt.close('all')


# plot Shannon diversity for matbench CV datasets
fig = plt.figure(figsize=(20,8))
avg_diversity = df_matbench_cv.groupby(by='mat_prop').agg({'train_diversity': 'mean'})['train_diversity']
std_diversity = df_matbench_cv.groupby(by='mat_prop').agg({'train_diversity': 'std'})['train_diversity']
y_pos = np.arange(len(avg_diversity))
plt.barh(y_pos, avg_diversity, xerr=std_diversity, tick_label=matbench_data)
ax_list = fig.axes
ax_list[0].invert_yaxis()
plt.title('Shannon diversity index for matbench datasets, across 5 CVs')
plt.savefig(f'{fig_path}/shannon_diversity_matbench_data.png', bbox_inches='tight', dpi=200)


# plot Shannon diversity for both datasets
fig = plt.figure(figsize=(18,12))
avg_diversity = df_matbench_cv.groupby(by='mat_prop').agg({'mat_prop':'first','train_diversity': 'mean'})
std_diversity = df_matbench_cv.groupby(by='mat_prop').agg({'mat_prop':'first','train_diversity': 'std'})
all_avg_diversity = pd.concat([avg_diversity, df_benchmark[['mat_prop', 'train_diversity']]], axis=0, ignore_index=True)
all_std_diversity = pd.concat([std_diversity, pd.concat([df_benchmark['mat_prop'], pd.DataFrame(np.full(df_benchmark.shape[0],fill_value=np.nan), columns=['train_diversity'])], axis=1)], axis=0, ignore_index=True)
labels = matbench_data + benchmark_data

color = ['#1f77b4' if l in matbench_data else '#9467bd' for l in labels]

y_pos = np.arange(len(all_avg_diversity))
plt.barh(y_pos, all_avg_diversity['train_diversity'],
         xerr=all_std_diversity['train_diversity'],
         color=color,
         tick_label=labels)
ax_list = fig.axes
ax_list[0].invert_yaxis()
ax_list[0].set_xlim(0.70, 1.01)
plt.title('Shannon diversity index for all datasets')
plt.savefig(f'{fig_path}/shannon_diversity_all_data.png', bbox_inches='tight', dpi=200)


plt.close('all')


# %%
df_benchmark['log2_samples_train'] = np.log2(df_benchmark['samples_train'].astype(float))
df_matbench_cv['log2_samples_train'] = np.log2(df_matbench_cv['samples_train'].astype(float))


# %%
# print diversity indices of the datasets
print(df_benchmark[['mat_prop', 'train_diversity']])
print(df_matbench_cv[['mat_prop', 'cv', 'train_diversity']])


# %%
# save dataset information to CSVs
df_benchmark.to_csv('data/df_benchmark.csv', index=False)
df_matbench_cv.to_csv('data/df_matbench_cv.csv', index=False)

