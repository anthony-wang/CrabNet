"""Train CrabNet using data for a given material property.

Use the crabnet environment.
"""
import os
from os.path import exists, join, dirname

# from warnings import warn
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from mat_discover.CrabNet.crabnet.kingcrab import CrabNet  # type: ignore
from mat_discover.CrabNet.crabnet.model import Model  # type: ignore

from mat_discover.CrabNet.utils.get_compute_device import get_compute_device

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def get_model(
    data_dir=join(dirname(__file__), "data", "materials_data"),
    mat_prop=None,
    train_df=None,
    val_df=None,
    test_df=None,
    classification=False,
    batch_size=None,
    transfer=None,
    verbose=True,
    losscurve=False,
    learningcurve=True,
    force_cpu=False,
    prefer_last=True,
):
    compute_device = get_compute_device(force_cpu=force_cpu, prefer_last=prefer_last)
    if train_df is None and val_df is None and test_df is None:
        if mat_prop is None:
            mat_prop = "example_materials_property"
        use_path = True
    else:
        if mat_prop is None:
            mat_prop = "DataFrame_property"
        use_path = False

    # Get the TorchedCrabNet architecture loaded
    model = Model(
        CrabNet(compute_device=compute_device).to(compute_device),
        model_name=f"{mat_prop}",
        verbose=verbose,
    )

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f"{transfer}.pth")
        model.model_name = f"{mat_prop}"

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    if use_path:
        # Get the datafiles you will learn from
        train_data = f"{data_dir}/{mat_prop}/train.csv"
        try:
            val_data = f"{data_dir}/{mat_prop}/val.csv"
        except IOError:
            print(
                "Please ensure you have train (train.csv) and validation data",
                f'(val.csv) in folder "data/materials_data/{mat_prop}"',
            )
        data_size = pd.read_csv(train_data).shape[0]
    else:
        train_data = train_df
        val_data = val_df
        data_size = train_data.shape[0]

    # Load the train and validation data before fitting the network
    batch_size = 2 ** round(np.log2(data_size) - 4)
    if batch_size < 2 ** 7:
        batch_size = 2 ** 7
    if batch_size > 2 ** 12:
        batch_size = 2 ** 12
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(
        f"training with batchsize {model.batch_size} "
        f"(2**{np.log2(model.batch_size):0.3f})"
    )
    if val_data is not None:
        model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=40, losscurve=losscurve, learningcurve=learningcurve)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ["composition", "target", "pred-0", "uncertainty"]
    if save_name is not None:
        save_path = "model_predictions"
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(f"{save_path}/{save_name}", index_label="Index")
    return df


def load_model(model, mat_prop, classification, data, verbose=True):
    # Load up a saved network.
    if type(model) is str:
        usepath = True
        model = Model(
            CrabNet(compute_device=compute_device).to(compute_device),
            model_name=f"{mat_prop}",
            verbose=verbose,
        )
        model_path = f"{mat_prop}.pth"
        model.load_network(model_path)
    else:
        usepath = False

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with

    if usepath and type(data) is str:
        data = f"{mat_prop}/{mat_prop}/{data}"
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2 ** 9, train=False)
    return model


def get_results(model):
    output = model.predict(loader=model.data_loader)  # predict the data saved here
    return model, output


def save_results(model, mat_prop, classification, data, verbose=True):
    if type(model) is str:
        usepath = True
        model = load_model(model, mat_prop, classification, data, verbose=verbose)
    else:
        usepath = False
        model.load_data(data, batch_size=2 ** 9, train=False)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f"{mat_prop} ROC AUC: {auc:0.3f}")
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f"{mat_prop} mae: {mae:0.3g}")

    # save predictions to a csv
    if usepath:
        fname = f'{mat_prop}_{data.replace(".csv", "")}_output.csv'
    else:
        fname = None
    df = to_csv(output, fname)
    return model, mae, df


def main(
    train_df=None,
    val_df=None,
    test_df=None,
    data_dir=join(dirname(__file__), "data", "materials_data"),
    mat_prop=None,
    classification=False,
    train=True,
    losscurve=False,
    learningcurve=True,
    verbose=True,
):
    """
    Train CrabNet model, save predictions, and output mean absolute error.

    Parameters
    ----------
    data_dir : TYPE, optional
        Directory where your data is stored.
        The default is 'data/materials_data'.
    mat_prop : TYPE, optional
        Folder with your materials properties.
        The default is 'example_materials_property'.
    classification : bool, optional
        Whether to perform classification. Performs regression if False.
        The default is False.
    train : bool, optional
        Whether to perform training. The default is True.

    Returns
    -------
    None.

    """
    if train_df is None and val_df is None and test_df is None:
        if mat_prop is None:
            mat_prop = "example_materials_property"
        train_data = join(data_dir, mat_prop, "train.csv")
        val_data = join(data_dir, mat_prop, "val.csv")
        test_data = join(data_dir, mat_prop, "test.csv")
        use_test = exists(test_data)
    else:
        mat_prop = "DataFrame_property"
        train_data = train_df
        val_data = val_df
        test_data = test_df
        use_test = test_df is not None

    # Train your model using the "get_model" function
    if train:
        if verbose:
            print(f'Property "{mat_prop}" selected for training')
        model = get_model(
            data_dir,
            mat_prop=mat_prop,
            classification=classification,
            verbose=verbose,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            losscurve=losscurve,
            learningcurve=learningcurve,
        )

    if verbose:
        cutter = "====================================================="
        first = " " * ((len(cutter) - len(mat_prop)) // 2) + " " * int(
            (len(mat_prop) + 1) % 2
        )
        last = " " * ((len(cutter) - len(mat_prop)) // 2)
        print("=====================================================")
        print(f"{first}{mat_prop}{last}")
        print("=====================================================")
        print("calculating train mae")
    model_train, mae_train, train_pred_df = save_results(
        model, mat_prop, classification, train_data, verbose=False
    )
    if val_data is not None:
        if verbose:
            print("-----------------------------------------------------")
            print("calculating val mae")
        model_val, mae_val, val_pred_df = save_results(
            model, mat_prop, classification, val_data, verbose=False
        )
    if use_test:
        if verbose:
            print("-----------------------------------------------------")
            print("calculating test mae")
        model_test, mae_test, test_pred_df = save_results(
            model, mat_prop, classification, test_data, verbose=False
        )
    else:
        test_pred_df = None
    if verbose:
        print("=====================================================")
    return train_pred_df, val_pred_df, test_pred_df


# %%
if __name__ == "__main__":
    main()

# %% Code Graveyard
# HACK:
# try:
#     from .crabnet.kingcrab import CrabNet
#     from .crabnet.model import Model
# except ImportWarning:
#     warn(
#         "relative import didn't work, probably because code is being executed as script instead of package",
#         ImportWarning,
#     )
