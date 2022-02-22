from warnings import warn
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# retrieve static file from package: https://stackoverflow.com/a/20885799/13697228
from importlib.resources import open_text

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

import torch
from torch.optim.lr_scheduler import CyclicLR

from .utils.get_compute_device import get_compute_device

from .utils.utils import (
    Lamb,
    Lookahead,
    RobustL1,
    RobustL2,
    BCEWithLogitsLoss,
    EDM_CsvLoader,
    Scaler,
    DummyScaler,
    count_parameters,
)
from crabnet.utils.optim import SWA

from crabnet.kingcrab import SubCrab

# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


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


def data(
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
        `from crabnet.data.materials_data import elasticity`
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


# %%
class Model:
    def __init__(
        self,
        model=None,
        model_name="UnnamedModel",
        n_elements="infer",
        verbose=True,
        force_cpu=False,
        prefer_last=True,
        fudge=0.02,
        out_dims=3,
        d_model=512,
        extend_features=None,
        d_extend=0,
        N=3,
        heads=4,
        elem_prop="mat2vec",
    ):
        """
        Model class for instantiating, training, and predicting with CrabNet models

        Parameters
        ----------
        model : _CrabNet Model, optional
            Specify existing CrabNet model to use, by default None
        model_name : str, optional
            The name of your model, by default "UnnamedModel"
        n_elements : str, optional
            The maximum number of elements to consider during featurization, by default "infer"
        verbose : bool, optional
            Whether model information and progress should be printed, by default True
        force_cpu : bool, optional
            Put all models on the cpu regardless of other available devices CPU, by default False
        prefer_last : bool, optional
            Whether to prefer last used compute_device, by default True
        fudge : float, optional
            The "fudge" (i.e. noise) applied to the fractional encodings, by default 0.02
        out_dims : int, optional
            Output dimensions for Residual Network, by default 3
        d_model : int, optional
            Size of the Model, see paper, by default 512
        extend_features : _type_, optional
            Whether extended features will be included, by default None
        d_extend : int, optional
            Number of extended features to include, by default 0
        N : int, optional
            Number of attention layers, by default 3
        heads : int, optional
            Number of attention heads to use, by default 4
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
        """
        if model is None:
            compute_device = get_compute_device(
                force_cpu=force_cpu, prefer_last=prefer_last
            )
            model = SubCrab(
                compute_device=compute_device,
                out_dims=out_dims,
                d_model=d_model,
                d_extend=d_extend,
                N=N,
                heads=heads,
            )
        self.model = model
        self.model_name = model_name
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.n_elements = n_elements
        self.compute_device = model.compute_device
        self.extend_features = extend_features
        self.fudge = fudge  #  expected fractional tolerance (std. dev) ~= 2%
        self.verbose = verbose
        self.elem_prop = elem_prop
        if self.verbose:
            print("\nModel architecture: out_dims, d_model, N, heads")
            print(
                f"{self.model.out_dims}, {self.model.d_model}, "
                f"{self.model.N}, {self.model.heads}"
            )
            print(f"Running on compute device: {self.compute_device}")
            print(f"Model size: {count_parameters(self.model)} parameters\n")

    def load_data(self, data, extra_features=None, batch_size=2**9, train=False):
        self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(
            data=data,
            extra_features=extra_features,
            batch_size=batch_size,
            n_elements=self.n_elements,
            inference=inference,
            verbose=self.verbose,
            elem_prop=self.elem_prop,
        )
        if self.verbose:
            print(
                f"loading data with up to {data_loaders.n_elements:0.0f} "
                f"elements in the formula"
            )

        # update n_elements after loading dataset
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        if train:
            self.train_len = len(y)
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def train(self):
        self.model.train()
        ti = time()
        minima = []
        for i, data in enumerate(self.train_loader):
            X, y, formula, extra_features = data
            y = self.scaler.scale(y)
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            # add a small jitter to the input fractions to improve model
            # robustness and to increase stability
            # frac = frac * (1 + (torch.rand_like(frac)-0.5)*self.fudge)  # uniform
            frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

            src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(
                self.compute_device, dtype=data_type_torch, non_blocking=True
            )
            y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
            extra_features = extra_features.to(
                self.compute_device, dtype=data_type_torch, non_blocking=True
            )
            output = self.model.forward(src, frac, extra_features=extra_features)
            prediction, uncertainty = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), uncertainty.view(-1), y.view(-1))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping:
                self.lr_scheduler.step()

            swa_check = self.epochs_step * self.swa_start - 1
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                act_v, pred_v, _, _ = self.predict(loader=self.data_loader)
                if np.any(np.isnan(pred_v)):
                    warn("NaN values found in `pred_v`. Replacing with zeros.")
                    pred_v = np.nan_to_num(pred_v)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)

        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            if self.verbose:
                print(f"Epoch {self.epoch} failed to improve.")
                print(
                    f"Discarded: {self.optimizer.discard_count}/"
                    f"{self.discard_n} weight updates"
                )

        dt = time() - ti
        datalen = len(self.train_loader.dataset)
        # print(f'training speed: {datalen/dt:0.3f}')

    def fit(
        self,
        epochs=None,
        checkin=None,
        losscurve=False,
        learningcurve=True,
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
    ):
        assert_train_str = "Please Load Training Data (self.train_loader)"
        assert_val_str = "Please Load Validation Data (self.data_loader)"
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve["train"] = []
        self.loss_curve["val"] = []

        self.epochs_step = epochs_step
        self.step_size = self.epochs_step * len(self.train_loader)
        if self.verbose:
            print(
                f"stepping every {self.step_size} training passes,",
                f"cycling lr every {self.epochs_step} epochs",
            )
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            if self.verbose:
                print(f"running for {epochs} epochs")
        if checkin is None:
            # TODO: not sure if epochs_step has to be 10 (and epochs=40) for this to
            # work (check with Anthony)
            checkin = self.epochs_step * 2
            if self.verbose:
                print(
                    f"checkin at {self.epochs_step*2} " f"epochs to match lr scheduler"
                )
        if epochs % (self.epochs_step * 2) != 0:
            updated_epochs = epochs - epochs % (self.epochs_step * 2)
            if self.verbose:
                print(
                    f"epochs not divisible by {self.epochs_step * 2}, "
                    f"updating epochs to {updated_epochs} for learning"
                )
            epochs = updated_epochs

        self.step_count = 0

        criterion_lookup = {
            "RobustL1": RobustL1,
            "RobustL2": RobustL2,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }

        if criterion is None:
            if self.classification:
                if self.verbose:
                    print("Using BCE loss for classification task")
                self.criterion = BCEWithLogitsLoss
            else:
                self.criterion = RobustL1
        elif type(criterion) is str:
            self.criterion = criterion_lookup[criterion]
        else:
            self.criterion = criterion

        base_optim = Lamb(
            params=self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adam=adam,
            min_trust=min_trust,
        )
        optimizer = Lookahead(base_optimizer=base_optim, alpha=alpha, k=k)
        # NOTE: SWA has hyperparameters, but not sure I want to deal with them now
        self.optimizer = SWA(optimizer)

        lr_scheduler = CyclicLR(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            cycle_momentum=False,
            step_size_up=self.step_size,
        )

        self.swa_start = 2  # start at (n/2) cycle (lr minimum)
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        # NOTE: parameter
        self.discard_n = 3

        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            ti = time()
            self.train()
            # print(f'epoch time: {(time() - ti):0.3f}')
            self.lr_list.append(self.optimizer.param_groups[0]["lr"])

            if (epoch + 1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                ti = time()
                act_t, pred_t, _, _ = self.predict(loader=self.train_loader)
                dt = time() - ti
                datasize = len(act_t)
                # print(f'inference speed: {datasize/dt:0.3f}')
                # PARAMETER: mae vs. rmse?
                if np.any(np.isnan(pred_t)):
                    warn("NaN values found in `pred_t`. Replacing with zeros.")
                    pred_t = np.nan_to_num(pred_t)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve["train"].append(mae_t)
                act_v, pred_v, _, _ = self.predict(loader=self.data_loader)
                if np.any(np.isnan(pred_v)):
                    warn("NaN values found in `pred_v`. Replacing with zeros.")
                    pred_v = np.nan_to_num(pred_v)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve["val"].append(mae_v)
                epoch_str = f"Epoch: {epoch}/{epochs} ---"
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f"train auc: {train_auc:0.3f}"
                    val_str = f"val auc: {val_auc:0.3f}"
                if self.verbose:
                    print(epoch_str, train_str, val_str)

                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch + 1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve["val"])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve["train"], "o-", label="train_mae")
                    plt.plot(xval, self.loss_curve["val"], "s--", label="val_mae")
                    plt.plot(
                        self.xswa, self.yswa, "o", ms=12, mfc="none", label="SWA point"
                    )
                    plt.ylim(0, 2 * np.mean(self.loss_curve["val"]))
                    plt.title(f"{self.model_name}")
                    plt.xlabel("epochs")
                    plt.ylabel("MAE")
                    plt.legend()
                    plt.show()

            if epoch == epochs - 1 or self.optimizer.discard_count >= self.discard_n:
                # save output df for stats tracking
                xval = np.arange(len(self.loss_curve["val"])) * checkin - 1
                xval[0] = 0
                tval = self.loss_curve["train"]
                vval = self.loss_curve["val"]
                os.makedirs("figures/lc_data", exist_ok=True)
                df_loss = pd.DataFrame([xval, tval, vval]).T
                df_loss.columns = ["epoch", "train loss", "val loss"]
                df_loss["swa"] = ["n"] * len(xval)
                df_loss.loc[df_loss["epoch"].isin(self.xswa), "swa"] = "y"
                df_loss.to_csv(f"figures/lc_data/{self.model_name}_lc.csv", index=False)

                # save output learning curve plot
                if learningcurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve["val"])) * checkin - 1
                    xval[0] = 0
                    plt.plot(
                        xval,
                        self.loss_curve["train"],
                        "o-",
                        label="$\mathrm{MAE}_\mathrm{train}$",
                    )
                    plt.plot(
                        xval,
                        self.loss_curve["val"],
                        "s--",
                        label="$\mathrm{MAE}_\mathrm{val}$",
                    )
                    if self.epoch >= (self.epochs_step * self.swa_start - 1):
                        plt.plot(
                            self.xswa,
                            self.yswa,
                            "o",
                            ms=12,
                            mfc="none",
                            label="SWA point",
                        )
                    plt.ylim(0, 2 * np.mean(self.loss_curve["val"]))
                    plt.title(f"{self.model_name}")
                    plt.xlabel("epochs")
                    plt.ylabel("MAE")
                    plt.legend()
                    plt.savefig(f"figures/lc_data/{self.model_name}_lc.png")
                    plt.show()

            if self.optimizer.discard_count >= self.discard_n:
                if self.verbose:
                    print(
                        f"Discarded: {self.optimizer.discard_count}/"
                        f"{self.discard_n} weight updates, "
                        f"early-stopping now"
                    )
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    def predict(self, data=None, loader=None):
        if data is None and loader is None:
            raise SyntaxError("Specify either data *or* loader, not neither.")
        elif data is not None and loader is None:
            if self.extend_features is not None:
                extra_features = data[self.extend_features]
            else:
                extra_features = None
            self.load_data(data, extra_features=extra_features)
            loader = self.data_loader
        elif data is not None and loader is not None:
            raise SyntaxError("Specify either data *or* loader, not both.")
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0]) / 2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula, extra_features = data
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(
                    self.compute_device, dtype=data_type_torch, non_blocking=True
                )
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                extra_features = extra_features.to(
                    self.compute_device, dtype=data_type_torch, non_blocking=True
                )
                output = self.model.forward(src, frac, extra_features=extra_features)
                prediction, uncertainty = output.chunk(2, dim=-1)
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i * self.batch_size, i * self.batch_size + len(y), 1)

                atoms[data_loc, :] = src.cpu().numpy()
                fractions[data_loc, :] = frac.cpu().numpy()
                act[data_loc] = y.view(-1).cpu().numpy()
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy()
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy()
                formulae[data_loc] = formula

        return (act, pred, formulae, uncert)

    def save_network(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
            os.makedirs("models/trained_models", exist_ok=True)
            path = f"models/trained_models/{model_name}.pth"
            if self.verbose:
                print(f"Saving network ({model_name}) to {path}")
        else:
            path = f"models/trained_models/{model_name}.pth"
            if self.verbose:
                print(f"Saving checkpoint ({model_name}) to {path}")

        self.network = {
            "weights": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "model_name": model_name,
        }
        torch.save(self.network, path)

    def load_network(self, model_data):
        if type(model_data) is str:
            path = f"models/trained_models/{model_data}"
            network = torch.load(path, map_location=self.compute_device)
        else:
            network = model_data
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(network["weights"])
        self.scaler.load_state_dict(network["scaler_state"])
        self.model_name = network["model_name"]


# %%
if __name__ == "__main__":
    pass
