#%%
"""Test CrabNet's fit and predict via `get_model()` and `predict()`."""
import numpy as np
from crabnet.model import data
from crabnet.data.materials_data import elasticity
from crabnet.train_crabnet import get_model

train_df, val_df = data(elasticity, dummy=True)


def test_crabnet():
    mdl = get_model(train_df=train_df, val_df=val_df, verbose=True)

    train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)
    return train_true, train_pred, formulas, train_sigma


def test_extend_features():
    train_df["state_var0"] = np.random.rand(train_df.shape[0])
    val_df["state_var0"] = np.random.rand(val_df.shape[0])
    mdl = get_model(
        train_df=train_df, val_df=val_df, extend_features=["state_var0"], verbose=True
    )

    train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)
    return train_true, train_pred, formulas, train_sigma


def test_extend_transfer():

    train_df["state_var0"] = np.random.rand(train_df.shape[0])
    val_df["state_var0"] = np.random.rand(val_df.shape[0])

    mdl = get_model(
        train_df=train_df,
        val_df=val_df,
        transfer=True,
        extend_transfer=False,
        extend_features=["state_var0"],
        verbose=True,
    )

    mdl.fit(transfer=True, extend_transfer=True)

    train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)
    return train_true, train_pred, formulas, train_sigma


if __name__ == "__main__":
    train_true, train_pred, formulas, train_sigma = test_extend_transfer()
    train_true, train_pred, formulas, train_sigma = test_crabnet()
    train_true, train_pred, formulas, train_sigma = test_extend_features()
    1 + 1

# %%
