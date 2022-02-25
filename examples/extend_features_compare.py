"""Compare CrabNet `extend_features` with XGBoost for hardness dataset.

Dependency: `pip install vickers_hardness`, but then might run into issue with shapely.
See fix here: https://github.com/uncertainty-toolbox/uncertainty-toolbox/issues/59

"""
from os.path import join
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from crabnet.utils.data import get_data
from crabnet.crabnet_ import CrabNet
import vickers_hardness.data as vh_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
)
from vickers_hardness.utils.plotting import parity_with_err
from vickers_hardness.vickers_hardness_ import VickersHardness

dummy = False
hyperopt = False
split_by_groups = True
remove_load = True

# %% directories
figure_dir = join("figures", "extend_features")
result_dir = join("results", "extend_features")

if remove_load:
    figure_dir = join(figure_dir, "without_load")
    result_dir = join(result_dir, "without_load")

crabnet_figures = join(figure_dir, "crabnet")
crabnet_results = join(result_dir, "crabnet")

xgboost_figures = join(figure_dir, "xgboost")
xgboost_results = join(result_dir, "xgboost")

for path in [crabnet_figures, crabnet_results, xgboost_figures, xgboost_results]:
    Path(path).mkdir(parents=True, exist_ok=True)

# %% load dataset
X = get_data(vh_data, "hv_des.csv", groupby=False, split=False).rename(
    columns={"composition": "formula"}
)
prediction = get_data(vh_data, "hv_comp_load.csv", groupby=False, split=False)
y = prediction["hardness"]
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.1)

if dummy:
    X = X.head(50)
    y = y.head(50)

if remove_load:
    X["load"] = np.zeros(X.shape[0])  # could try np.random.rand

# %% K-fold cross-validation
if split_by_groups:
    ss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    cv = GroupKFold()
    cvtype = "gcv"
else:
    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    cv = KFold(shuffle=True, random_state=100)  # ignores groups
    cvtype = "cv"

if split_by_groups:
    groups = X["formula"]
else:
    groups = None

trainval_idx, test_idx = list(ss.split(X, y, groups=groups))[0]
X_test, y_test = X.iloc[test_idx, :], y[test_idx]
X, y = X.iloc[trainval_idx, :], y.iloc[trainval_idx]

if split_by_groups:
    subgroups = X["formula"]
else:
    subgroups = None

crabnet_dfs = []
xgb_dfs = []
for train_index, test_index in cv.split(X, y, subgroups):
    X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    train_df = pd.DataFrame(
        {"formula": X_train["formula"], "load": X_train["load"], "target": y_train}
    )
    val_df = pd.DataFrame(
        {"formula": X_val["formula"], "load": X_val["load"], "target": y_val}
    )
    cb = CrabNet(
        extend_features=["load"],
        verbose=True,
        learningcurve=False,
    )
    cb.fit(train_df)
    y_true, y_pred, formulas, y_std = cb.predict(val_df)
    crabnet_dfs.append(
        pd.DataFrame(
            {
                "actual_hardness": y_true,
                "predicted_hardness": y_pred,
                "y_std": y_std,
                "load": val_df["load"],
                "formula": val_df["formula"],
            }
        )
    )

    vickers = VickersHardness(hyperopt=hyperopt)
    vickers.fit(X_train, y_train)
    y_pred, y_std = vickers.predict(X_val, y_val, return_uncertainty=True)
    xgb_dfs.append(
        pd.DataFrame(
            {
                "actual_hardness": val_df["target"],
                "predicted_hardness": y_pred,
                "y_std": y_std,
                "load": val_df["load"],
                "formula": val_df["formula"],
            }
        )
    )

crabnet_df = pd.concat(crabnet_dfs)
xgb_df = pd.concat(xgb_dfs)

parity_with_err(
    crabnet_df,
    error_y="y_std",
    figfolder=crabnet_figures,
    fname=f"parity_stderr_{cvtype}",
)
parity_with_err(
    xgb_df,
    error_y="y_std",
    figfolder=xgboost_figures,
    fname=f"parity_stderr_{cvtype}",
)

names: List[str] = ["crabnet", "xgboost"]
for name in names:
    if name == "crabnet":
        tmp_df = crabnet_df
    elif name == "xgboost":
        tmp_df = xgb_df
    else:
        raise NotImplementedError(f"{name} not implemented")

    y_true, y_pred = [tmp_df["actual_hardness"], tmp_df["predicted_hardness"]]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{name} MAE: {mae:.5f}")
    print(f"{name} RMSE: {rmse:.5f}")
    tmp_df.sort_index().to_csv(join(result_dir, name, f"{cvtype}-results.csv"))


# %% results
## with applied load
# crabnet MAE: 3.06177
# crabnet RMSE: 5.12390
# xgboost MAE: 2.34908
# xgboost RMSE: 3.81564

## without applied load
# crabnet MAE: 4.42722
# crabnet RMSE: 6.24093
# xgboost MAE: 3.96865
# xgboost RMSE: 5.22576
