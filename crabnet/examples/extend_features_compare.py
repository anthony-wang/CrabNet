"""Compare CrabNet `extend_features` with XGBoost for hardness dataset.


Dependency: `pip install vickers_hardness`, but then might run into issue with shapely.
See fix here: https://github.com/uncertainty-toolbox/uncertainty-toolbox/issues/59

"""
from os.path import join

import numpy as np
import pandas as pd
from crabnet.model import data
import vickers_hardness.data as vh_data
from crabnet.train_crabnet import get_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    cross_validate,
)
from vickers_hardness.utils.plotting import parity_with_err
from vickers_hardness.vickers_hardness_ import VickersHardness

hyperopt = False
split_by_groups = False

# %% load dataset
X = data(vh_data, "hv_des.csv", groupby=False, split=False).rename(
    columns={"composition": "formula"}
)
prediction = data(vh_data, "hv_comp_load.csv", groupby=False, split=False)
y = prediction["hardness"]
# X, X_test, y, y_test = train_test_split(X, y, test_size=0.1)


# %% K-fold cross-validation
if split_by_groups:
    ss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    cv = GroupKFold()
    cvtype = "gcv"
    groups = X["formula"]
else:
    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    cv = KFold(shuffle=True, random_state=100)  # ignores groups
    cvtype = "cv"
    groups = None

trainval_idx, test_idx = list(ss.split(X, y, groups=groups))[0]
X_test, y_test = X.iloc[test_idx, :], y[test_idx]
X, y = X.iloc[trainval_idx, :], y[trainval_idx]

results = cross_validate(
    VickersHardness(hyperopt=hyperopt),
    X,
    y,
    groups=groups,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_estimator=True,
)

for train_index, test_index in cv.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_df = pd.DataFrame(
        {"formula": X_train["formula"], "load": X_train["load"], "target": y_train}
    )
    val_df = pd.DataFrame(
        {"formula": X_train["formula"], "load": X_train["load"], "target": y_train}
    )
    mdl = get_model(
        train_df=train_df, val_df=val_df, extend_features=["state_var0"], verbose=True
    )

train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)

estimators = results["estimator"]
result_dfs = [estimator.result_df for estimator in estimators]
merge_df = pd.concat(result_dfs)
merge_df["actual_hardness"] = y

parity_with_err(
    merge_df, error_y="y_upper", error_y_minus="y_lower", fname=f"parity_ci_{cvtype}"
)
parity_with_err(merge_df, error_y="y_std", fname=f"parity_stderr_{cvtype}")
parity_with_err(merge_df, fname=f"parity_stderr_calib_{cvtype}")

y_true, y_pred = [merge_df["actual_hardness"], merge_df["predicted_hardness"]]
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

merge_df.sort_index().to_csv(join("results", f"{cvtype}-results.csv"))
1 + 1


# %% Code Graveyard

