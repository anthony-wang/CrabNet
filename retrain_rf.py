import os
import numpy as np
import pandas as pd

from time import time
import matplotlib.pyplot as plt

from ast import literal_eval

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from joblib import dump, load

from utils.utils import get_cbfv
from utils.utils import CONSTANTS

plt.rcParams.update({"font.size": 16})


# %%
cons = CONSTANTS()

bm_names = cons.benchmark_props
bm_pretty = cons.benchmark_names_dict
mb_names = cons.matbench_props
mb_pretty = cons.matbench_names_dict
elem_props = cons.eps
elem_props = ["magpie"]

mat_props = mb_names + bm_names

model_names = {
    "RandomForestRegressor": RandomForestRegressor,
}


# %%
def concatenate_best_results(directory):
    files = os.listdir(directory)

    csv_list = [file for file in files if ".csv" in file if "best_" in file]
    csv_paths = [os.path.join(directory, file) for file in csv_list]

    df = pd.DataFrame()

    for file in csv_paths:
        df_file = pd.read_csv(file)
        df = pd.concat([df, df_file], axis=0, ignore_index=True)

    df = df[
        [
            "estimator",
            "elem_prop",
            "mat_prop",
            "params",
            "mean_test_neg_MAE",
            "mean_test_r2",
        ]
    ]

    return df


def get_best_model(df, mat_prop, sortby="mean_test_neg_MAE"):
    df_mp = df[df["mat_prop"] == mat_prop]
    df_best_row = df_mp.sort_values(sortby, axis=0, ascending=False).iloc[0]

    params = df_best_row["params"]
    params = literal_eval(params)

    est_name = df_best_row["estimator"]
    est_class = model_names[est_name]
    estimator = est_class(**params)

    elem_prop = df_best_row["elem_prop"]

    return est_name, estimator, params, elem_prop


# %%
if __name__ == "__main__":
    metrics_path = r"metrics/rf_gridsearch/"

    bm_predictions_path = r"metrics/rf_benchmark__predictions/"
    mb_predictions_path = r"metrics/rf_matbench__predictions/"
    clf_save_path = f"models/trained_rf/"
    os.makedirs(clf_save_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(bm_predictions_path, exist_ok=True)
    os.makedirs(mb_predictions_path, exist_ok=True)

    bm_props_dir = r"data/benchmark_data/"
    mb_props_dir = r"data/matbench_cv/"

    df_best_classics = concatenate_best_results(metrics_path)

    retrain_columns = ["estimator_name", "estimator", "params", "mat_prop", "elem_prop"]
    df_models_to_retrain = pd.DataFrame(columns=retrain_columns)

    for mp in mat_props:
        output = get_best_model(df_best_classics, mp)
        est_name, estimator, params, elem_prop = output

        row_dict = {
            "estimator_name": est_name,
            "estimator": estimator,
            "params": params,
            "mat_prop": mp,
            "elem_prop": elem_prop,
        }
        df_models_to_retrain = df_models_to_retrain.append(row_dict, ignore_index=True)

    df_retrain_columns = [
        "estimator_name",
        "fit_time",
        "mae_train",
        "mse_train",
        "r2_train",
        "mae_test",
        "mse_test",
        "r2_test",
        "params",
        "mat_prop",
        "elem_prop",
    ]
    df_retrain_results = pd.DataFrame(columns=df_retrain_columns)
    ti_retrain_classics = time()

    mb_cvs = 5
    for i in range(len(df_models_to_retrain)):
        output = df_models_to_retrain.iloc[i].values
        est_name, estimator, param, mat_prop, elem_prop = output

        if mat_prop in bm_names:
            df_predictions = pd.DataFrame()
            trainpath = os.path.join(bm_props_dir, mat_prop, "train.csv")
            valpath = os.path.join(bm_props_dir, mat_prop, "val.csv")
            testpath = os.path.join(bm_props_dir, mat_prop, "test.csv")

            X, y, form, skipped = get_cbfv(trainpath, elem_prop=elem_prop)
            X_val, y_val, form_val, skipped_val = get_cbfv(valpath, elem_prop=elem_prop)
            X_test, y_test, form_test, skipped_test = get_cbfv(
                testpath, elem_prop=elem_prop
            )

            X = pd.concat([X, X_val], axis=0, ignore_index=True)
            y = pd.concat([y, y_val], axis=0, ignore_index=True)
            form = pd.concat([form, form_val], axis=0, ignore_index=True)

            print(f"fitting {mat_prop} with {elem_prop} using {est_name}")
            if X.shape[0] > 2000:
                print(
                    f"there are {X.shape[0]} data samples, " f"this may take a while..."
                )
            ti_model = time()
            estimator.fit(X, y)
            dt_model = time() - ti_model
            print(f"finished fitting {mat_prop} with {elem_prop} using {est_name}")
            print(f"fitting time: {dt_model:0.4f} s")

            savepath = f"models/trained_rf/{mat_prop}.joblib"
            print(f"Saving rf model {savepath}")
            dump(estimator, savepath, compress=3, protocol=5)

            print(f"evaluating fitted model on whole trainset and testset")
            ti_eval = time()
            target_test = y_test
            pred_test = estimator.predict(X_test)

            target_train = y
            pred_train = estimator.predict(X)

            r2_test = r2_score(target_test, pred_test)
            mae_test = mean_absolute_error(target_test, pred_test)
            mse_test = mean_squared_error(target_test, pred_test)

            r2_train = r2_score(target_train, pred_train)
            mae_train = mean_absolute_error(target_train, pred_train)
            mse_train = mean_squared_error(target_train, pred_train)

            dt_eval = time() - ti_eval
            print(f"evaluation time: {dt_eval:0.4f} s")

            df_retrain_row = {
                "estimator_name": est_name,
                "fit_time": dt_model,
                "mae_train": mae_train,
                "mse_train": mse_train,
                "r2_train": r2_train,
                "mae_test": mae_test,
                "mse_test": mse_test,
                "r2_test": r2_test,
                "params": param,
                "mat_prop": mat_prop,
                "cv": "NaN",
                "elem_prop": elem_prop,
            }

            df_retrain_results = df_retrain_results.append(
                df_retrain_row, ignore_index=True
            )

            df_predictions["composition"] = form_test
            df_predictions["target"] = target_test
            df_predictions["pred"] = pred_test
            outfile = os.path.join(bm_predictions_path, f"{mat_prop}_test_output.csv")
            df_predictions.to_csv(outfile)
            print(f"mae_test for {mat_prop}: {mae_test}")

        elif mat_prop in mb_names:
            trainpath_pre = os.path.join(mb_props_dir, mat_prop, "train")
            valpath_pre = os.path.join(mb_props_dir, mat_prop, "val")
            testpath_pre = os.path.join(mb_props_dir, mat_prop, "test")

            maes = []
            for cv in range(mb_cvs):
                df_predictions = pd.DataFrame()
                trainpath = f"{trainpath_pre}{cv}.csv"
                valpath = f"{valpath_pre}{cv}.csv"
                testpath = f"{testpath_pre}{cv}.csv"

                X, y, form, skipped = get_cbfv(trainpath, elem_prop=elem_prop)
                X_val, y_val, form_val, skipped_val = get_cbfv(
                    valpath, elem_prop=elem_prop
                )
                X_test, y_test, form_test, skipped_test = get_cbfv(
                    testpath, elem_prop=elem_prop
                )

                X = pd.concat([X, X_val], axis=0, ignore_index=True)
                y = pd.concat([y, y_val], axis=0, ignore_index=True)
                form = pd.concat([form, form_val], axis=0, ignore_index=True)

                print(f"fitting {mat_prop} with {elem_prop} using {est_name}")
                print(f"cv {cv}")
                if X.shape[0] > 2000:
                    print(
                        f"there are {X.shape[0]} data samples, "
                        f"this may take a while..."
                    )
                ti_model = time()
                estimator.fit(X, y)
                dt_model = time() - ti_model
                print(f"finished fitting {mat_prop} with {elem_prop} using {est_name}")
                print(f"fitting time: {dt_model:0.4f} s")

                savepath = f"models/trained_rf/{mat_prop}_cv{cv}.joblib"
                print(f"Saving rf model {savepath}")
                dump(estimator, savepath, compress=3, protocol=5)

                print(f"evaluating fitted model on whole trainset and testset")
                ti_eval = time()
                target_test = y_test
                pred_test = estimator.predict(X_test)

                target_train = y
                pred_train = estimator.predict(X)

                r2_test = r2_score(target_test, pred_test)
                mae_test = mean_absolute_error(target_test, pred_test)
                mse_test = mean_squared_error(target_test, pred_test)

                r2_train = r2_score(target_train, pred_train)
                mae_train = mean_absolute_error(target_train, pred_train)
                mse_train = mean_squared_error(target_train, pred_train)

                dt_eval = time() - ti_eval
                print(f"evaluation time: {dt_eval:0.4f} s")

                df_retrain_row = {
                    "estimator_name": est_name,
                    "fit_time": dt_model,
                    "mae_train": mae_train,
                    "mse_train": mse_train,
                    "r2_train": r2_train,
                    "mae_test": mae_test,
                    "mse_test": mse_test,
                    "r2_test": r2_test,
                    "params": param,
                    "mat_prop": mat_prop,
                    "cv": cv,
                    "elem_prop": elem_prop,
                }

                df_retrain_results = df_retrain_results.append(
                    df_retrain_row, ignore_index=True
                )

                df_predictions["composition"] = form_test
                df_predictions["target"] = target_test
                df_predictions["pred"] = pred_test
                outfile = os.path.join(
                    mb_predictions_path, f"{mat_prop}_test_output_cv{cv}.csv"
                )
                df_predictions.to_csv(outfile)
                print(f"mae_test: {mae_test}")
                maes.append(mae_test)
            print(f"average mae_test for {mat_prop}: {np.mean(maes)}")

    dt_retrain_classics = time() - ti_retrain_classics
    print("*********** retrain_classics finished ***********")
    print(f"retrain_classics finished, elapsed time: " f"{dt_retrain_classics:0.4g} s")
    print("*********** retrain_classics finished ***********")

    outfile = os.path.join(metrics_path, "retrained_test_scores.csv")
    df_retrain_results.to_csv(outfile, index=False)
