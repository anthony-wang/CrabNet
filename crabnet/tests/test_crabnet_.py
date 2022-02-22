"""Test CrabNet's fit and predict via `get_model()` and `predict()`."""
import numpy as np
from crabnet.model import data
from crabnet.data.materials_data import elasticity
from crabnet.crabnet_ import CrabNet

train_df, val_df = data(elasticity, dummy=True)


def test_crabnet():
    cb = CrabNet(
        compute_device="cpu", verbose=True, losscurve=False, learningcurve=False
    )
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


def test_extend_features():
    train_df["state_var0"] = np.random.rand(train_df.shape[0])
    val_df["state_var0"] = np.random.rand(val_df.shape[0])
    cb = CrabNet(verbose=True, extend_features=["state_var0"])
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


if __name__ == "__main__":
    train_pred, train_sigma = test_crabnet()
    train_pred, train_sigma = test_extend_features()
    1 + 1
