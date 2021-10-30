"""Test CrabNet's fit and predict via `get_model()` and `predict()`."""
from crabnet.model import data
from crabnet.data.materials_data import elasticity
from crabnet.train_crabnet import get_model


def test_crabnet():
    train_df, val_df = data(elasticity)
    mdl = get_model(train_df=train_df, val_df=val_df)

    train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)
    return train_true, train_pred, formulas, train_sigma


if __name__ == "__main__":
    train_true, train_pred, formulas, train_sigma = test_crabnet()
