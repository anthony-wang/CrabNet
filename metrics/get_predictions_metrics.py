import pandas as pd
import os
from utils.composition import _fractional_composition


def norm_form(formula):
    comp = _fractional_composition(formula)
    form = ""
    for key, value in comp.items():
        form += f"{key}{str(value)[0:9]}"
    return form


def count_elems(string):
    count = 0
    switch = 1
    for c in string:
        if c.isalpha():
            count += switch
            switch = 0
        if c.isnumeric():
            switch = 1
    return count


# %%
if __name__ == "__main__":
    print("processing all model predictions and calculating metrics")
    print("this will take a few minutes...")

    # %%
    results_path = "publication_predictions"
    benchmark_path = "data/benchmark_data"
    test_directories = os.listdir(results_path)
    benchmark_props = os.listdir(benchmark_path)

    benchmark_test_directories = [
        test for test in test_directories if "benchmark" in test
    ]

    dataset_results = {}
    dataset_preds = {}
    dataset_acts = {}
    test_maes = pd.DataFrame()
    df_stats = pd.DataFrame()
    for benchmark in benchmark_props:
        df_compositions = pd.DataFrame()
        df_preds = pd.DataFrame()
        df_acts = pd.DataFrame()
        models = []
        for directory in benchmark_test_directories:
            df_train_orig = pd.read_csv(
                f"{benchmark_path}/{benchmark}/train.csv",
                keep_default_na=False,
                na_values=[""],
            )
            df_val = pd.read_csv(
                f"{benchmark_path}/{benchmark}/val.csv",
                keep_default_na=False,
                na_values=[""],
            )
            df_train = pd.concat([df_train_orig, df_val], ignore_index=True)
            df_train["formula"] = [
                norm_form(formula) for formula in df_train["formula"]
            ]
            df_train.index = df_train["formula"]
            files = os.listdir(f"{results_path}\{directory}")
            file = [file for file in files if benchmark in file and "test" in file]
            if len(file) > 0:
                models.append(directory.split("_")[0])
                file = file[0]
                df = pd.read_csv(
                    f"{results_path}\{directory}\{file}",
                    keep_default_na=False,
                    na_values=[""],
                )
                composition = df["formula"]
                pred = df["predicted"]
                act = df["actual"]
                print(f"processing {benchmark} {models[-1]}")
                df_compositions = pd.concat([df_compositions, composition], axis=1)
                df_preds = pd.concat([df_preds, pred], axis=1)
                df_acts = pd.concat([df_acts, act], axis=1)

                n_total = act.count() + df_val.shape[0] + df_train_orig.shape[0]
                df_stats.at[benchmark, "mean_test"] = act.mean()
                df_stats.at[benchmark, "std_test"] = act.std()
                df_stats.at[benchmark, "n_test"] = act.count()

                df_stats.at[benchmark, "mean_train"] = df_train["target"].mean()
                df_stats.at[benchmark, "std_train"] = df_train["target"].std()
                df_stats.at[benchmark, "n_train"] = df_train_orig.shape[0]

                df_stats.at[benchmark, "n_val"] = df_val.shape[0]
                df_stats.at[benchmark, "n_total"] = n_total

                df_stats.at[benchmark, "prop_train"] = df_train_orig.shape[0] / n_total
                df_stats.at[benchmark, "prop_val"] = df_val.shape[0] / n_total
                df_stats.at[benchmark, "prop_test"] = act.count() / n_total

        df_compositions.columns = models
        df_preds.columns = models
        df_acts.columns = models
        df_diff = df_preds - df_acts
        df_mae = df_diff.abs().mean()
        test_maes[benchmark] = df_mae

        dataset_results[benchmark] = df_compositions
        dataset_preds[benchmark] = df_preds
        dataset_acts[benchmark] = df_acts

    maes = test_maes.T
    model_names = ["roost", "mat2vec", "onehot", "elemnet", "rf"]
    out_1 = maes[model_names]
    out = pd.concat([out_1, df_stats], axis=1)
    df_benchmark = out.copy()

    # %%
    results_path = "publication_predictions"
    matbench_path = "data/matbench_cv"
    test_directories = os.listdir(results_path)
    matbench_props = os.listdir(matbench_path)

    matbench_test_directories = [
        test for test in test_directories if "matbench" in test
    ]

    dataset_results = {}
    dataset_preds = {}
    dataset_acts = {}
    test_maes = pd.DataFrame()
    df_stats = pd.DataFrame()
    for matbench in matbench_props:
        df_compositions = pd.DataFrame()
        df_preds = pd.DataFrame()
        df_acts = pd.DataFrame()
        models = []
        for directory in matbench_test_directories:
            train_files = os.listdir(f"{matbench_path}/{matbench}")
            train_files = [file for file in train_files if "train" in file]
            test_files = os.listdir(f"{results_path}/{directory}")
            test_files = [
                file for file in test_files if matbench in file and "test" in file
            ]
            for i, (train_file, test_file) in enumerate(zip(train_files, test_files)):
                df_train_orig = pd.read_csv(
                    f"{matbench_path}/{matbench}/{train_file}",
                    keep_default_na=False,
                    na_values=[""],
                )
                df_val = pd.read_csv(
                    f'{matbench_path}/{matbench}/{train_file.replace("train", "val")}',
                    keep_default_na=False,
                    na_values=[""],
                )
                df_train = pd.concat([df_train_orig, df_val], ignore_index=True)
                df_train["formula"] = [
                    norm_form(formula) for formula in df_train["formula"]
                ]
                df_train.index = df_train["formula"]
                if len(file) > 0:
                    models.append(directory.split("_")[0] + f"_{i}")
                    file = file[0]
                    df = pd.read_csv(
                        f"{results_path}\{directory}\{test_file}",
                        keep_default_na=False,
                        na_values=[""],
                    )
                    df.index = df["formula"].values
                    composition = df["formula"]
                    pred = df["predicted"]
                    act = df["actual"]
                    print(f"processing {matbench} {models[-1]}")
                    df_compositions = pd.concat([df_compositions, composition], axis=1)
                    df_preds = pd.concat([df_preds, pred], axis=1)
                    df_acts = pd.concat([df_acts, act], axis=1)

                    n_total = act.count() + df_val.shape[0] + df_train_orig.shape[0]
                    df_stats.at[matbench, "mean_test"] = act.mean()
                    df_stats.at[matbench, "std_test"] = act.std()
                    df_stats.at[matbench, "n_test"] = act.count()

                    df_stats.at[matbench, "mean_train"] = df_train["target"].mean()
                    df_stats.at[matbench, "std_train"] = df_train["target"].std()
                    df_stats.at[matbench, "n_train"] = df_train_orig.shape[0]

                    df_stats.at[matbench, "n_val"] = df_val.shape[0]
                    df_stats.at[matbench, "n_total"] = n_total

                    df_stats.at[matbench, "prop_train"] = (
                        df_train_orig.shape[0] / n_total
                    )
                    df_stats.at[matbench, "prop_val"] = df_val.shape[0] / n_total
                    df_stats.at[matbench, "prop_test"] = act.count() / n_total

        df_compositions.columns = models
        df_preds.columns = models
        df_acts.columns = models
        df_diff = df_preds - df_acts
        df_mae_cv = df_diff.abs()
        df_mae = pd.DataFrame()
        model_names = []
        _ = [model_names.append(x[:-2]) for x in models if x[:-2] not in model_names]
        for j, model in enumerate(model_names):
            df_mae.loc[model, 0] = (
                df_mae_cv.iloc[:, (j) * 5 : (j + 1) * 5].max(axis=1).mean()
            )

        test_maes[matbench] = df_mae[0]
        dataset_results[matbench] = df_compositions
        dataset_preds[matbench] = df_preds
        dataset_acts[matbench] = df_acts

    maes = test_maes.T
    model_names = ["roost", "mat2vec", "onehot", "elemnet", "rf"]
    out_1 = maes[model_names]
    out = pd.concat([out_1, df_stats], axis=1)
    df_matbench = out.copy()

    # %%
    df_all = pd.concat([df_benchmark, df_matbench], axis=0, ignore_index=False)

    rename_dict = {"mat2vec": "crabnet", "onehot": "hotcrab"}
    df_all = df_all.rename(columns=rename_dict)
    df_all.to_csv("metrics/all_metrics.csv", index_label="property")
