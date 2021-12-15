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

from crabnet.kingcrab import CrabNet  # type: ignore
from crabnet.model import Model  # type: ignore

from crabnet.utils.get_compute_device import get_compute_device

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
    save=True,
):
    """Get a CrabNet model with default parameters set.
    #TODO: flesh out descriptions of parameters, as well as feasible min/max bounds
    where appropriate

    Parameters
    ----------
    data_dir : str, optional
        data directory, by default join(dirname(__file__), "data", "materials_data")
    mat_prop : str, optional
        name of material property (doesn't affect computation), by default None
    train_df : DataFrame, optional
        Training DataFrame with formula and target columns, by default None
    val_df : DataFrame, optional
        Validation DataFrame with formula and target columns (target can be dummy
        values, e.g. 0.0), by default None
    test_df : DataFrame, optional
        Test DataFrame with formula and target columns (OK if only train_df and val_df
        are specified), by default None
    classification : bool, optional
        Whether to perform classification. If False, then use regression, by default False
    batch_size : int, optional
        batch size for training the neural network, by default None
    transfer : str, optional
        Name of model to use for transfer learning, by default None
    verbose : bool, optional
        Whether to print verbose model information during the run, by default True
    losscurve : bool, optional
        Whether to plot a loss curve, by default False
    learningcurve : bool, optional
        Whether to plot a learning curve, by default True
    force_cpu : bool, optional
        Whether to force use of CPU. If False, then if compatible GPU is available, GPU is used, by default False
    prefer_last : bool, optional
        Whether to prefer the last used device (i.e. CPU or GPU), by default True
    fudge : float, optional
        The "fudge" (i.e. noise) applied to the fractional encodings, by default 0.02
    d_model : int, optional
        Model size. See paper, by default 512
    out_dims : int, optional
        [description], by default 3
    N : int, optional
        [description], by default 3
    heads : int, optional
        Number of attention heads to use, by default 4
    out_hidden : list, optional
        Hidden output dimensions of neural network, by default [1024, 512, 256, 128]
    emb_scaler : float, optional
        Embedding scaler. Value to multiply the embedding (`x`) by, by default 1.0
    pos_scaler : float, optional
        Positional scaler. Value to multiply the prevalance encoding (`pe`) by, by default 1.0
    pos_scaler_log : float, optional
        Positional log scaler. Value to multiply the prevalence log encoding (`ple`) by, by default 1.0
    bias : bool, optional
        Whether to bias nn.Linear, by default False
    dim_feedforward : int, optional
        [description], by default 2048
    dropout : float, optional
        [description], by default 0.1
    elem_prop : str, optional
        Which elemental feature vector to use. Possible values are "jarvis", "magpie",
        "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
    pe_resolution : int, optional
        Number of discretizations for the prevalence encoding, by default 5000
    ple_resolution : int, optional
        Number of discretizations for the prevalence log encoding, by default 5000
    epochs : int, optional
        How many epochs to allow the neural network to run for, by default 40
    epochs_step : int, optional
        [description], by default 10
    criterion : torch.nn Module, optional
        Or in other words the loss function (e.g. BCEWithLogitsLoss for classification
        or RobustL1 for regression), by default None. Possible values are
        `BCEWithLogitsLoss`, `RobustL1`, and `RobustL2`.
    lr : float, optional
        Learning rate, by default 1e-3
    betas : tuple, optional
        [description], by default (0.9, 0.999)
    eps : [type], optional
        [description], by default 1e-6
    weight_decay : int, optional
        [description], by default 0
    adam : bool, optional
        Whether to constrain the Lamb model to be the Adam model, by default False
    min_trust : [type], optional
        [description], by default None
    alpha : float, optional
        [description], by default 0.5
    k : int, optional
        [description], by default 6
    base_lr : float, optional
        Base learning rate, by default 1e-4
    max_lr : float, optional
        Max learning rate, by default 6e-3
    save : bool, optional
        Whether to save the network or not, by default True.

    Returns
    -------
    model
        instantiated CrabNet model
    """
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
        CrabNet(
            compute_device=compute_device,
            out_dims=out_dims,
            d_model=d_model,
            N=N,
            heads=heads,
            out_hidden=out_hidden,
            pe_resolution=pe_resolution,
            ple_resolution=ple_resolution,
            elem_prop=elem_prop,
            bias=bias,
            emb_scaler=emb_scaler,
            pos_scaler=pos_scaler,
            pos_scaler_log=pos_scaler_log,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(compute_device),
        model_name=f"{mat_prop}",
        verbose=verbose,
        fudge=fudge,
        out_dims=out_dims,
        d_model=d_model,
        N=N,
        heads=heads,
        elem_prop=elem_prop,
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
    if batch_size is None:
        batch_size = 2 ** round(np.log2(data_size) - 4)
        if batch_size < 2 ** 7:
            batch_size = 2 ** 7
        if batch_size > 2 ** 12:
            batch_size = 2 ** 12
    model.load_data(train_data, batch_size=batch_size, train=True)
    if verbose:
        print(
            f"training with batchsize {model.batch_size} "
            f"(2**{np.log2(model.batch_size):0.3f})"
        )
    if val_data is not None:
        model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(
        epochs=epochs,
        losscurve=losscurve,
        learningcurve=learningcurve,
        epochs_step=epochs_step,
        criterion=criterion,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        adam=adam,
        min_trust=min_trust,
        alpha=alpha,
        k=k,
        base_lr=base_lr,
        max_lr=max_lr,
    )

    # Save the network (saved as f"{model_name}.pth")
    if save:
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
    model.load_data(data, batch_size=2 ** 9, train=False, verbose=verbose)
    return model


def get_results(model, verbose=True):
    output = model.predict(
        loader=model.data_loader, verbose=verbose
    )  # predict the data saved here
    return model, output


def save_results(model, mat_prop, classification, data, verbose=False):
    if type(model) is str:
        usepath = True
        model = load_model(model, mat_prop, classification, data, verbose=verbose)
    else:
        usepath = False
        model.load_data(data, batch_size=2 ** 9, train=False, verbose=verbose)
    model, output = get_results(model, verbose=verbose)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        if verbose:
            print(f"{mat_prop} ROC AUC: {auc:0.3f}")
    else:
        mae = np.abs(output[0] - output[1]).mean()
        if verbose:
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
        if isinstance(val_data, pd.DataFrame):
            val_data[val_data.target.isnull()] = 0
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
