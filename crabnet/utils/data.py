from importlib.resources import open_text

import pandas as pd
from sklearn.model_selection import train_test_split


def groupby_formula(df, how="max", mapper=None):
    """Group identical compositions together and preserve original indices.

    See https://stackoverflow.com/a/49216427/13697228

    Parameters
    ----------
    df : DataFrame
        At minimum should contain "formula" and "target" columns.
    how : str, optional
        How to perform the "groupby", either "mean" or "max", by default "max"

    Returns
    -------
    DataFrame
        The grouped DataFrame such that the original indices are preserved.
    """
    if mapper is not None:
        df = df.rename(columns=mapper)
    grp_df = (
        df.reset_index()
        .groupby(by="formula")
        .agg({"index": lambda x: tuple(x), "target": how})
        .reset_index()
    )
    return grp_df


def get_data(
    module,
    fname="train.csv",
    mapper=None,
    groupby=True,
    dummy=False,
    split=True,
    val_size=0.2,
    test_size=0.0,
    random_state=42,
):
    """Grab data from within the subdirectories (modules) of mat_discover.

    Parameters
    ----------
    module : Module
        The module within CrabNet that contains e.g. "train.csv". For example,
        `from CrabNet.data.materials_data import elasticity`
    fname : str, optional
        Filename of text file to open.
    mapper: dict, optional
        Column renamer for pandas DataFrame (i.e. used in `df.rename(columns=mapper)` By default, None.
    dummy : bool, optional
        Whether to pare down the data to a small test set, by default False
    groupby : bool, optional
        Whether to use groupby_formula to filter identical compositions
    split : bool, optional
        Whether to split the data into train, val, and (optionally) test sets, by default True
    val_size : float, optional
        Validation dataset fraction, by default 0.2
    test_size : float, optional
        Test dataset fraction, by default 0.0
    random_state : int, optional
        seed to use for the train/val/test split, by default 42

    Returns
    -------
    DataFrame
        If split==False, then the full DataFrame is returned directly

    DataFrame, DataFrame
        If test_size == 0 and split==True, then training and validation DataFrames are returned.

    DataFrame, DataFrame, DataFrame
        If test_size > 0 and split==True, then training, validation, and test DataFrames are returned.
    """
    train_csv = open_text(module, fname)
    df = pd.read_csv(train_csv)

    if groupby:
        df = groupby_formula(df, how="max", mapper=mapper)

    if dummy:
        ntot = min(100, len(df))
        df = df.head(ntot)

    if split:
        if test_size > 0:
            df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )

        train_df, val_df = train_test_split(
            df, test_size=val_size / (1 - test_size), random_state=random_state
        )

        if test_size > 0:
            return train_df, val_df, test_df
        else:
            return train_df, val_df
    else:
        return df
