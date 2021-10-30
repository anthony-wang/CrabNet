import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.composition import _element_composition


# %%
benchmark_data = os.listdir("data/benchmark_data")
matbench_data = os.listdir("data/matbench_cv")


# %%
cols = [
    "mat_prop",
    "samples_total",
    "samples_train",
    "samples_val",
    "samples_test",
    "prop_train",
    "prop_val",
    "prop_test",
]
df_benchmark = pd.DataFrame(columns=cols)

for prop in tqdm(benchmark_data, desc="Processing benchmark_data"):
    df_train = pd.read_csv(
        f"data/benchmark_data/{prop}/train.csv", keep_default_na=False, na_values=[""]
    )
    df_val = pd.read_csv(
        f"data/benchmark_data/{prop}/val.csv", keep_default_na=False, na_values=[""]
    )
    df_test = pd.read_csv(
        f"data/benchmark_data/{prop}/test.csv", keep_default_na=False, na_values=[""]
    )
    n_train = df_train["formula"].apply(_element_composition).apply(len).max()
    n_val = df_val["formula"].apply(_element_composition).apply(len).max()
    n_test = df_test["formula"].apply(_element_composition).apply(len).max()
    n_elements = max([n_train, n_val, n_test])
    total_samples = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
    df_row = {
        "mat_prop": prop,
        "n_elements": n_elements,
        "samples_total": total_samples,
        "samples_train": df_train.shape[0],
        "samples_val": df_val.shape[0],
        "samples_test": df_test.shape[0],
        "prop_train": df_train.shape[0] / total_samples,
        "prop_val": df_val.shape[0] / total_samples,
        "prop_test": df_test.shape[0] / total_samples,
    }
    df_benchmark = df_benchmark.append(df_row, ignore_index=True)


# %%
cols = [
    "mat_prop",
    "samples_total",
    "samples_train",
    "samples_val",
    "samples_test",
    "prop_train",
    "prop_val",
    "prop_test",
]
df_matbench_cv = pd.DataFrame(columns=cols)

for prop in tqdm(matbench_data, desc="Processing matbench_data"):
    df_train = pd.read_csv(
        f"data/matbench_cv/{prop}/train0.csv", keep_default_na=False, na_values=[""]
    )
    df_val = pd.read_csv(
        f"data/matbench_cv/{prop}/val0.csv", keep_default_na=False, na_values=[""]
    )
    df_test = pd.read_csv(
        f"data/matbench_cv/{prop}/test0.csv", keep_default_na=False, na_values=[""]
    )
    n_train = df_train["formula"].apply(_element_composition).apply(len).max()
    n_val = df_val["formula"].apply(_element_composition).apply(len).max()
    n_test = df_test["formula"].apply(_element_composition).apply(len).max()
    n_elements = max([n_train, n_val, n_test])
    total_samples = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
    df_row = {
        "mat_prop": prop,
        "n_elements": n_elements,
        "samples_total": total_samples,
        "samples_train": df_train.shape[0],
        "samples_val": df_val.shape[0],
        "samples_test": df_test.shape[0],
        "prop_train": df_train.shape[0] / total_samples,
        "prop_val": df_val.shape[0] / total_samples,
        "prop_test": df_test.shape[0] / total_samples,
    }
    df_matbench_cv = df_matbench_cv.append(df_row, ignore_index=True)


# %%
# cols = ['mat_prop', 'samples_total']
# df_matbench_all = pd.DataFrame(columns=cols)

# for prop in tqdm(matbench_data, desc="Processing matbench_data"):
#     df_data = pd.read_csv(f'data/matbench/{prop}.csv',
#                           keep_default_na=False, na_values=[''])
#     n_elements = df_data['formula'].apply(_element_composition).apply(len).max()
#     df_row = {
#         'mat_prop': prop,
#         'n_elements': n_elements,
#         'samples_total': df_data.shape[0],
#         }
#     df_matbench_all = df_matbench_all.append(df_row, ignore_index=True)


# %%
df_benchmark["log2_samples_train"] = (
    df_benchmark["samples_train"].astype(float).apply(np.log2)
)
df_matbench_cv["log2_samples_train"] = (
    df_matbench_cv["samples_train"].astype(float).apply(np.log2)
)
# df_matbench_all['log2_samples_total'] = df_matbench_all['samples_total'].astype(float).apply(np.log2)


# %%
print(df_benchmark.to_latex(index=False, escape=True, float_format="{:0.2f}".format))

# %%
print(df_matbench_cv.to_latex(index=False, escape=True, float_format="{:0.2f}".format))

# %%
# print(df_matbench_all.to_latex(index=False, escape=True, float_format="{:0.2f}".format))


# %%
df_benchmark
df_matbench_cv
# df_matbench_all


# %%
df_benchmark.to_csv("df_benchmark.csv")
df_matbench_cv.to_csv("df_matbench_cv.csv")
