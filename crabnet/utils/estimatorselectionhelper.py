import warnings

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

plt.rcParams.update({"font.size": 16})


# %%
class EstimatorSelectionHelper:
    def __init__(self, models, params, random_seed=42):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(f"Missing estimator parameters: {missing_params}")

    def fit(
        self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False, random_seed=42
    ):
        if n_jobs == 1:
            warnings.warn(
                "n_jobs is currently set at 1, "
                "consider passing n_jobs=-1 or n_jobs=n_cores, "
                "where n_cores is the number of CPU cores you have",
                RuntimeWarning,
            )
        for key in self.keys:
            print(f"\nRunning GridSearchCV for {key}.")
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(
                model,
                params,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
            )
            gs.fit(X, y)
            self.grid_searches[key] = gs
        self.scoring = scoring
        self.scoring_val2key = {scoring[k]: k for k in scoring}

    def plot_gridsearch(self, model_name, elem_prop, mat_prop, fig_dir, gs):
        test_metric = "mean_test_neg_MAE"

        # Get CV parameters and results
        dims = pd.DataFrame(gs.cv_results_["params"])
        # for RandomForest: convert max_depth = None to string 'None'
        dims.loc[dims["max_depth"].isnull(), "max_depth"] = "None"
        dims["score"] = gs.cv_results_[test_metric]
        col_names = dims.columns.tolist()
        dims = dims.pivot(col_names[0], col_names[1], col_names[2])
        dims_flipped = dims.iloc[::-1]

        # Plot and save gridsearch heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(dims_flipped)
        plt.title(
            f"{test_metric}\n" f"{model_name} on {elem_prop} with {mat_prop}",
            fontsize=18,
        )
        plt.savefig(
            fig_dir + f"{model_name}_{elem_prop}_{mat_prop}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close("all")
        print(f"saved figure {model_name}_{elem_prop}_{mat_prop}.png")

    def score_summary(self, ep, mp, fig_dir, sort_by="mean_test_r2"):
        print("***************** gridsearch done *****************")
        scoring_keys = self.scoring.keys()

        col_prefixes = ["mean_", "std_"]
        col_midfixes = ["train_", "test_"]
        col_suffixes = scoring_keys
        columns1 = ["estimator"]
        columns2 = [
            p + m + s for s in col_suffixes for m in col_midfixes for p in col_prefixes
        ]
        all_columns = (
            columns1 + columns2 + ["params"] + ["mean_fit_time", "mean_score_time"]
        )

        df = pd.DataFrame(columns=all_columns)
        df_best_models = pd.DataFrame(columns=all_columns)

        # Plot gridsearch results for each model/param/elem_prop/mat_prop
        # combination, and save dataframe of best results
        for m in self.grid_searches:
            self.plot_gridsearch(m, ep, mp, fig_dir, self.grid_searches[m])

            df_model = pd.DataFrame(columns=all_columns)
            print(f"Parsing results for {m}")
            res = self.grid_searches[m].cv_results_
            params = res["params"]
            df_model["estimator"] = [m] * len(params)
            for col in df_model.columns:
                if col != "estimator":
                    df_model[col] = res[col]

            best_run_idx = df_model["mean_test_neg_MAE"].idxmax()
            best_run = df_model.loc[[best_run_idx], :]
            df_best_models = pd.concat([df_best_models, best_run], axis=0)
            df = pd.concat([df, df_model], axis=0)

        df["elem_prop"] = ep
        df["mat_prop"] = mp
        df_best_models["elem_prop"] = ep
        df_best_models["mat_prop"] = mp

        return df, df_best_models
