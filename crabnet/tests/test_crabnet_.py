"""Test CrabNet's fit and predict via `get_model()` and `predict()`."""
import numpy as np
from crabnet.utils.data import get_data
from crabnet.data.materials_data import elasticity
from crabnet.crabnet_ import CrabNet
import pandas as pd
from matbench.bench import MatbenchBenchmark
from time import time

train_df, val_df = get_data(elasticity, dummy=True)


def test_crabnet():
    cb = CrabNet(
        compute_device="cpu",
        verbose=True,
        losscurve=False,
        learningcurve=False,
        epochs=40,
    )
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


def test_crabnet_300_epochs():
    cb = CrabNet(
        compute_device="cpu",
        verbose=True,
        losscurve=False,
        learningcurve=False,
        epochs=300,
    )
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


def test_extend_features():
    train_df["state_var0"] = np.random.rand(train_df.shape[0])
    val_df["state_var0"] = np.random.rand(val_df.shape[0])
    cb = CrabNet(verbose=True, extend_features=["state_var0"], epochs=40)
    cb.fit(train_df)
    train_pred, train_sigma = cb.predict(val_df, return_uncertainty=True)
    return train_pred, train_sigma


def test_matbench_expt_gap():
    t0 = time()
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

    for task in mb.tasks:
        task.load()
        for fold in [task.folds[0]]:

            # Inputs are either chemical compositions as strings
            # or crystal structures as pymatgen.Structure objects.
            # Outputs are either floats (regression tasks) or bools (classification tasks)
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            train_df = pd.DataFrame({"formula": train_inputs, "target": train_outputs})

            # Get testing data
            test_inputs = task.get_test_data(fold, include_target=False)
            test_df = pd.DataFrame(
                {"formula": test_inputs, "target": np.zeros(test_inputs.shape[0])}
            )

            # adjust parameters to have it run quickly on CPU
            crab = CrabNet(
                epochs=80,
                d_model=64,
                batch_size=256,
                heads=2,
                out_hidden=list(np.array([1024, 512, 256, 128]) // 16),
                dim_feedforward=2048 // 16,
                pe_resolution=1000,
                ple_resolution=1000,
                learningcurve=False,
                losscurve=False,
                compute_device="cpu",
            )
            crab.fit(train_df)
            predictions = crab.predict(test_df)

            # Record your data!
            task.record(fold, predictions)

            mae = task.results["fold_0"]["scores"]["mae"]
            tol = 0.42
            print(f"mae: {mae}")
            assert mae < tol, f"mae ({mae}) should be less than {tol} for this fold"
    tf = time()
    print(f"Elapsed time (s): {tf - t0}")


# def test_extend_transfer():

#     train_df["state_var0"] = np.random.rand(train_df.shape[0])
#     val_df["state_var0"] = np.random.rand(val_df.shape[0])

#     mdl = get_model(
#         train_df=train_df,
#         val_df=val_df,
#         extend_transfer=False,
#         extend_features=["state_var0"],
#         verbose=True,
#     )

#     mdl.fit(extend_transfer=True)

#     train_true, train_pred, formulas, train_sigma = mdl.predict(val_df)
#     return train_true, train_pred, formulas, train_sigma

if __name__ == "__main__":
    test_matbench_expt_gap()
    train_pred, train_sigma = test_crabnet_300_epochs()
    train_pred, train_sigma = test_crabnet()
    train_pred, train_sigma = test_extend_features()
    1 + 1
