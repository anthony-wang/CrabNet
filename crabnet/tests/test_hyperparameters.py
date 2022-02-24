"""Test CrabNet's fit and predict via `get_model()` and `predict()`."""
from crabnet.utils.data import get_data
from crabnet.data.materials_data import elasticity
from crabnet.crabnet_ import CrabNet

train_df, val_df = get_data(elasticity, dummy=True)


def test_hyperparameters():
    cb = CrabNet(
        batch_size=None,
        fudge=0.02,
        d_model=512,
        out_dims=3,
        N=3,
        heads=4,
        out_hidden=[1024, 512, 256, 128],
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        bias=False,
        dim_feedforward=2048,
        dropout=0.1,
        elem_prop="mat2vec",
        pe_resolution=5000,
        ple_resolution=5000,
        epochs=40,
        epochs_step=10,
        criterion=None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
        min_trust=None,
        alpha=0.5,
        k=6,
        base_lr=1e-4,
        max_lr=6e-3,
    )
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


if __name__ == "__main__":
    train_pred, train_sigma = test_hyperparameters()
