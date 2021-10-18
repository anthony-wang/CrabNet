import os

from time import time

from utils.utils import get_cbfv

from utils.estimatorselectionhelper import EstimatorSelectionHelper


# %%
def modelselectionhelper(
    models,
    params,
    elem_props,
    mat_props_dir,
    mat_props,
    metrics_dir,
    fig_dir,
    scoring=None,
    n_jobs=1,
    cv=3,
    refit="neg_MAE",
    verbose=False,
    random_seed=42,
):
    if scoring is None:
        scoring = {"neg_MAE": "neg_mean_absolute_error"}
    for ep in elem_props:
        print("\n++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"currently using element property: {ep}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        ti_ep = time()

        for mp in mat_props:
            print(f"fitting {mp} with {ep} using {cv}-fold CV")
            ti_mp = time()

            trainpath = os.path.join(mat_props_dir, mp, "train.csv")
            valpath = os.path.join(mat_props_dir, mp, "val.csv")

            if not os.path.exists(trainpath) or not os.path.exists(valpath):
                trainpath = os.path.join(mat_props_dir, mp, "train0.csv")
                valpath = os.path.join(mat_props_dir, mp, "val0.csv")

            X, y, form, skipped = get_cbfv(trainpath, elem_prop=ep)
            X_val, y_val, form_val, skipped_val = get_cbfv(valpath, elem_prop=ep)

            # Sample the dataset for faster gridsearch
            n_samples = 2000
            if X.shape[0] > n_samples:
                print(
                    f"Sampling training data to {n_samples} samples "
                    f"to speed up initial gridsearch"
                )
                X = X.sample(n=n_samples)
                y = y.loc[X.index]
                form = form.loc[X.index]

            helper1 = EstimatorSelectionHelper(models, params)
            helper1.fit(
                X,
                y,
                scoring=scoring,
                n_jobs=n_jobs,
                cv=cv,
                refit=refit,
                verbose=verbose,
                random_seed=random_seed,
            )

            output = helper1.score_summary(ep, mp, fig_dir, sort_by="mean_test_r2")
            score_summary, best_models = output

            print("\n************************************************")
            print(f"finished {mp} with {ep}")
            print("saving score summary and best models files")
            print("************************************************")

            outpath_all = os.path.join(metrics_dir, f"{ep}_{mp}.csv")
            score_summary.to_csv(outpath_all, index=False)

            outpath_best = os.path.join(metrics_dir, f"best_{ep}_{mp}.csv")
            best_models.to_csv(outpath_best, index=False)

            dt_mp = time() - ti_mp
            print(f"time elapsed for {mp} with {ep}: {dt_mp:0.4f} s")

        dt_ep = time() - ti_ep
        print(
            f"time elapsed for all material properties " f"using {ep}: {dt_ep:0.4f} s"
        )

    return helper1
