"""Top-level module for instantiating a CrabNet model to predict properties."""
import os
from os import PathLike
from os.path import dirname, join
from typing import Callable, List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import CyclicLR

from crabnet.kingcrab import SubCrab

# for backwards compatibility of imports
from crabnet.utils.data import get_data, groupby_formula  # noqa: F401
from crabnet.utils.optim import SWA

from crabnet.utils.get_compute_device import get_compute_device
from crabnet.utils.utils import (
    BCEWithLogitsLoss,
    DummyScaler,
    EDM_CsvLoader,
    Lamb,
    Lookahead,
    RobustL1,
    RobustL2,
    Scaler,
    count_parameters,
)

# retrieve static file from package: https://stackoverflow.com/a/20885799/13697228


# %%
class CrabNet(nn.Module):
    """Model class for instantiating, training, and predicting with CrabNet models."""

    def __init__(
        self,
        model: Optional[Union[str, SubCrab]] = None,
        model_name: str = "UnnamedModel",
        n_elements: Union[str, int] = "infer",
        classification: bool = False,
        verbose: bool = True,
        force_cpu: bool = False,
        prefer_last: bool = True,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        epochs_step: int = 10,
        checkin: Optional[int] = None,
        fudge: float = 0.02,
        out_dims: int = 3,
        d_model: int = 512,
        extend_features: Optional[List[str]] = None,
        N: int = 3,
        heads: int = 4,
        elem_prop: str = "mat2vec",
        compute_device: Optional[Union[str, torch.device]] = None,
        out_hidden: List[int] = [1024, 512, 256, 128],
        pe_resolution: int = 5000,
        ple_resolution: int = 5000,
        bias=False,
        emb_scaler: float = 1.0,
        pos_scaler: float = 1.0,
        pos_scaler_log: float = 1.0,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        val_size: float = 0.2,
        criterion: Optional[Union[str, Callable]] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        adam: bool = False,
        min_trust: Optional[float] = None,
        alpha: float = 0.5,
        k: int = 6,
        base_lr: float = 1e-4,
        max_lr: float = 6e-3,
        random_state: Optional[int] = None,
        mat_prop: Optional[Union[str, PathLike]] = None,
        losscurve: bool = True,
        learningcurve: bool = True,
        save: bool = True,
    ):
        """
        Instantiate a CrabNet model.

        Parameters
        ----------
        model : _CrabNet
            Instantiated CrabNet class, by default None.
        model_name : str, optional
            The name of your model, by default "UnnamedModel"
        n_elements : str, optional
            The maximum number of elements to consider during featurization, by default
            "infer"
        classification : bool, optional
            Whether to perform classification. If False, then assume regression. By
            default, False
        verbose : bool, optional
            Whether model information and progress should be printed, by default True
        force_cpu : bool, optional
            Put all models on the cpu regardless of other available devices CPU, by default False
        prefer_last : bool, optional
            Whether to prefer last used compute_device, by default True
        batch_size : int
            The batch size to use during training. If not None, then used as-is. If
            specified, then it is assigned either 2 ** 7 == 128 or 2 ** 12 == 4096
            based on the value of `data_size`.
        epochs : int, optional
            How many epochs (# of passes through entire dataset). If None, then this is
            automatically assigned based on the dataset size using
            `get_epochs_checkin_stepsize`. The number of epochs must be even. By default
            None
        checkin : int, optional
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None
        fudge : float, optional
            The "fudge" (i.e. noise) applied to the fractional encodings, by default 0.02
        out_dims : int, optional
            Output dimensions for Residual Network, by default 3
        d_model : int, optional
            Size of the Model, see paper, by default 512
        extend_features : _type_, optional
            Whether extended features will be included, by default None
        N : int, optional
            Number of attention layers, by default 3
        heads : int, optional
            Number of attention heads to use, by default 4
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default
            "mat2vec"
        compute_device : _type_, optional
            Computing device to run model on, by default None
        out_hidden : list(int), optional
            Architecture of hidden layers in the Residual Network, by default [1024, 512, 256, 128]
        pe_resolution : int, optional
            Number of discretizations for the prevalence encoding, by default 5000
        ple_resolution : int, optional
            Number of discretizations for the prevalence log encoding, by default 5000
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
        bias : bool, optional
            Whether to bias the Residual Network, by default False
        emb_scaler : float, optional
            Float value by which to scale the elemental embeddings, by default 1.0
        pos_scaler : float, optional
            Float value by which to scale the fractional encodings, by default 1.0
        pos_scaler_log : float, optional
            Float value by which to scale the log fractional encodings, by default 1.0
        dim_feedforward : int, optional
            Dimenions of the feed forward network following transformer, by default 2048
        dropout : float, optional
            Percent dropout in the feed forward network following the transformer, by
            default 0.1
        val_size : float, optional
            fraction of validation data to take from training data only if `val_df` is
            None. By default, 0.2
        criterion : torch.nn Module, optional
            Or in other words the loss function (e.g. BCEWithLogitsLoss for classification
            or RobustL1 for regression), by default None. Possible values are
            `BCEWithLogitsLoss`, `RobustL1`, and `RobustL2`.
        lr : float, optional
            Learning rate, by default 1e-3
        betas : tuple, optional
            Coefficients on gradient and squared gradient during ``Lamb`` optimization, by default (0.9, 0.999)
        eps : float, optional
            Value added to the denominator during ``Lamb`` optimization, by default 1e-6
        weight_decay : float, optional
            L2 penalty in ``Lamb``, by default 0
        adam : bool, optional
            Whether to constrain the ``Lamb`` model to be the Adam model, by default False
        min_trust : float, optional
            [description], by default None
        alpha : float, optional
            ``Lookahead`` "slow update" rate, by default 0.5
        k : int, optional
            Number of ``Lookahead`` steps, by default 6
        base_lr : float, optional
            Base learning rate, by default 1e-4
        max_lr : float, optional
            Max learning rate, by default 6e-3
        random_state : int, optional
            The seed to use for both `torch` and `numpy` random number generators. If
            None, then this has no effect. By default None.
        mat_prop : str, optional
            name of material property (doesn't affect computation), by default None
        losscurve : bool, optional
            Whether to plot a loss curve, by default False
        learningcurve : bool, optional
            Whether to plot a learning curve, by default True
        save : bool, optional
            Whether to save the weights of the model, by default True
        """
        super().__init__()
        if compute_device is None:
            compute_device = get_compute_device(
                force_cpu=force_cpu, prefer_last=prefer_last
            )
        elif compute_device == "cpu":
            compute_device = torch.device("cpu")
        self.compute_device = compute_device

        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.extend_features = extend_features
        self.N = N
        self.heads = heads
        self.compute_device = compute_device
        self.bias = bias
        self.out_hidden = out_hidden

        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_step = epochs_step
        self.checkin = checkin

        self.pe_resolution = pe_resolution
        self.ple_resolution = ple_resolution
        self.emb_scaler = emb_scaler
        self.pos_scaler = pos_scaler
        self.pos_scaler_log = pos_scaler_log
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.criterion = criterion
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adam = adam
        self.min_trust = min_trust
        self.alpha = alpha
        self.k = k
        self.base_lr = base_lr
        self.max_lr = max_lr

        # Apply BCEWithLogitsLoss to model output if binary classification is True
        if classification:
            self.classification = True

        self.model_name = model_name
        self.mat_prop = mat_prop
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.n_elements = n_elements

        self.fudge = fudge  #  expected fractional tolerance (std. dev) ~= 2%
        self.verbose = verbose
        self.elem_prop = elem_prop

        self.losscurve = losscurve
        self.learningcurve = learningcurve
        self.losscurve_fig = None
        self.learningcurve_fig = None

        self.val_size = val_size
        self.model = model
        self.save = save

        self.criterion_lookup = {
            "RobustL1": RobustL1,
            "RobustL2": RobustL2,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        self.data_type_torch = torch.float32

        if self.verbose:
            print("\nModel architecture: out_dims, d_model, N, heads")
            print(f"{self.out_dims}, {self.d_model}, " f"{self.N}, {self.heads}")
            print(f"Running on compute device: {self.compute_device}")

    def fit(
        self,
        train_df: pd.DataFrame = None,
        val_df: pd.DataFrame = None,
        extend_features: List[str] = None,
        data_dir: Union[str, PathLike] = join(
            dirname(__file__), "data", "materials_data"
        ),
        transfer: str = None,
    ):
        """Fit CrabNet to training data and update hyperparams with validation data.

        Parameters
        ----------
        train_df, val_df : pd.DataFrame, optional
            Training and validation data with at minimum, "formula" and "target" columns
            and optionally, "extra features" (based on names in `extend_features`). If
            `val_df` is None, then `test_size` determines the amount of training data to
            be split into `val_df`. By default None
        extend_features : List[str], optional
            Names of columns to use as extra features from `train_df` and `val_df`, by default None
        data_dir : str, optional
            The directory from which to load data if loading from a file rather than
            a DataFrame. `data_dir` is only used if both `train_df` and `val_df` are
            None. It is assumed that the files in the data directory will be named
            `train.csv`, `val.csv`, and `test.csv`. By default join(dirname(__file__), "data", "materials_data")
        transfer : str, optional
            Path to the saved weights to use for transfer learning. If None, then no
            transfer learning is performed. By default None
        """
        self.d_extend = 0 if extend_features is None else len(extend_features)

        if self.model is None:
            self.model = SubCrab(
                compute_device=self.compute_device,
                out_dims=self.out_dims,
                d_model=self.d_model,
                d_extend=self.d_extend,
                N=self.N,
                heads=self.heads,
                pe_resolution=self.pe_resolution,
                ple_resolution=self.ple_resolution,
                emb_scaler=self.emb_scaler,
                pos_scaler=self.pos_scaler,
                pos_scaler_log=self.pos_scaler_log,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            ).to(self.compute_device)

        if self.verbose:
            print(f"Model size: {count_parameters(self.model)} parameters\n")

        # self.transfer_nn = TransferNetwork(512, 512)

        # Train network starting at pretrained weights
        if transfer is not None:
            self.load_network(f"{transfer}.pth")
            self.model_name = f"{self.mat_prop}"

        (
            train_data,
            val_data,
            data_size,
            extra_train_data,
            extra_val_data,
        ) = self._separate_extended_features(train_df, val_df, data_dir)

        self.batch_size = self._default_batch_size(self.batch_size, data_size)

        assert isinstance(self.batch_size, int)
        self._load_trainval_data(
            self.batch_size, train_data, val_data, extra_train_data, extra_val_data
        )

        self.epochs, self.checkin, self.stepsize = self._get_epochs_checkin_stepsize(
            self.epochs, self.checkin
        )
        assert isinstance(self.epochs, int)
        assert isinstance(self.checkin, int)

        self.step_count = 0

        self._select_criterion(self.criterion)
        assert self.criterion is not None

        assert isinstance(self.model, SubCrab)
        base_optim = Lamb(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            adam=self.adam,
            min_trust=self.min_trust,
        )
        optimizer = Lookahead(base_optimizer=base_optim, alpha=self.alpha, k=self.k)
        self.optimizer = SWA(optimizer)

        lr_scheduler = CyclicLR(
            self.optimizer,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            cycle_momentum=False,
            step_size_up=self.stepsize,
        )
        self.lr_scheduler = lr_scheduler

        self.loss_curve: dict = {"train": [], "val": []}

        self.stepping = True
        self.swa_start = 2  # start at (n/2) cycle (lr minimum)
        self.xswa: List[int] = []
        self.yswa: List[float] = []

        self.lr_list: List[float] = []
        self.discard_n = 3

        assert isinstance(self.epochs, int)
        assert isinstance(self.checkin, int)
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.epochs = self.epochs
            self._train()
            self.lr_list.append(self.optimizer.param_groups[0]["lr"])

            if (
                (epoch + 1) % self.checkin == 0
                or epoch == self.epochs - 1
                or epoch == 0
            ):
                self._losscurve_stats(self.epochs, epoch)

                if self.losscurve:
                    self._plot_losscurve(self.checkin)

            self._track_stats(self.epochs, self.checkin, self.learningcurve, epoch)

            if self.optimizer.discard_count >= self.discard_n:
                if self.verbose:
                    print(
                        f"Discarded: {self.optimizer.discard_count}/{self.discard_n}weight updates, early-stopping now"
                    )
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

        if self.save:
            self.save_network()

    def predict(
        self,
        test_df: pd.DataFrame = None,
        loader=None,
        return_uncertainty=False,
        return_true=False,
    ):
        """Predict on new data using a fitted CrabNet model.

        Parameters
        ----------
        test_df : pd.DataFrame, optional
            _description_, by default None
        loader : torch.Dataloader, optional
            The Dataloader corresponding to the test data, by default None
        return_uncertainty : bool, optional
            Whether to return standard deviation uncertainties. If `return_true`, then
            `return_uncertainty` takes precendence and is returned as the second output. By default False
        return_true : bool, optional
            Whether to return the true values (used for comparison with the predicted
            values). If `return_uncertainty` is also specified, then the uncertainties
            appear before the true values (i.e. pred, std, true), by default False

        Returns
        -------
        pred : np.array
            Predicted values. Always returned.
        uncert : np.array
            Standard deviation uncertainty. Returned if `return_uncertainty`. Precedes
            `act` if `act` is also returned.
        act : np.array
            True values. Returned if `return_true`. `uncert` precedes `act` if both
            `uncert` and `act` are returned.

        Raises
        ------
        SyntaxError
            "Specify either data *or* loader, not neither."
        SyntaxError
            "Specify either data *or* loader, not both."
        """
        if test_df is None and loader is None:
            raise SyntaxError("Specify either data *or* loader, not neither.")
        elif test_df is not None and loader is None:
            if self.extend_features is not None:
                extra_features = test_df[self.extend_features]
            else:
                extra_features = None
            self.load_data(test_df, extra_features=extra_features)
            loader = self.data_loader
        elif test_df is not None and loader is not None:
            raise SyntaxError("Specify either data *or* loader, not both.")

        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0]) / 2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        assert isinstance(self.model, SubCrab)
        self.model.eval()

        with torch.no_grad():
            for i, batch_df in enumerate(loader):
                # extract data
                X, y, formula, extra_features = batch_df
                src, frac = X.squeeze(-1).chunk(2, dim=1)

                # send to device
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(
                    self.compute_device, dtype=self.data_type_torch, non_blocking=True
                )
                y = y.to(
                    self.compute_device, dtype=self.data_type_torch, non_blocking=True
                )
                extra_features = extra_features.to(
                    self.compute_device, dtype=self.data_type_torch, non_blocking=True
                )

                # predict
                output = self.model.forward(src, frac, extra_features=extra_features)
                prediction, uncertainty = output.chunk(2, dim=-1)
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                assert self.batch_size is not None
                data_loc = slice(i * self.batch_size, i * self.batch_size + len(y), 1)

                atoms[data_loc, :] = src.cpu().numpy()
                fractions[data_loc, :] = frac.cpu().numpy()
                act[data_loc] = y.view(-1).cpu().numpy()
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy()
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy()
                formulae[data_loc] = formula

        if return_uncertainty and return_true:
            return pred, uncert, act
        elif return_uncertainty and not return_true:
            return pred, uncert
        elif not return_uncertainty and return_true:
            return pred, act
        else:
            return pred

    def load_data(
        self,
        data: Union[str, pd.DataFrame],
        extra_features: pd.DataFrame = None,
        batch_size: int = 2**9,
        train: bool = False,
    ):
        """Load data using PyTorch Dataloader.

        Parameters
        ----------
        data : Union[str, pd.DataFrame]
            The data to load, which can be the folder in which the ``.csv`` file resides
            or a pandas DataFrame.
        extra_features : pd.DataFrame, optional
            DataFrame containing the extra features that will be used for training (e.g.
            state variables) that were extracted based on the column names in `CrabNet().extend_features`, by default None
        batch_size : int, optional
            The batch size to use during training. By default 2 ** 9
        train : bool, optional
            Whether this is the training data, by default False
        """
        if self.batch_size is None:
            self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(
            data=data,
            extra_features=extra_features,
            batch_size=self.batch_size,
            n_elements=self.n_elements,
            inference=inference,
            verbose=self.verbose,
            elem_prop=self.elem_prop,
        )
        if self.verbose:
            print(
                f"loading data with up to {data_loaders.n_elements:0.0f} elements in the formula"
            )

        # update n_elements after loading dataset
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        if train:
            self.train_len = len(y)
            if self.classification:
                self.scaler: Union[Scaler, DummyScaler] = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def _train(self):
        """Train the SubCrab PyTorch model using backpropagation."""
        minima = []
        for data in self.train_loader:
            # separate into src and frac
            X, y, _, extra_features = data
            y = self.scaler.scale(y)
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            frac = self._add_jitter(src, frac)

            # send to PyTorch device
            src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(
                self.compute_device, dtype=self.data_type_torch, non_blocking=True
            )
            y = y.to(self.compute_device, dtype=self.data_type_torch, non_blocking=True)
            extra_features = extra_features.to(
                self.compute_device, dtype=self.data_type_torch, non_blocking=True
            )

            # train
            output = self.model.forward(src, frac, extra_features=extra_features)
            prediction, uncertainty = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), uncertainty.view(-1), y.view(-1))

            # backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping:
                self.lr_scheduler.step()

            # hyperparameter updates
            swa_check = self.epochs_step * self.swa_start - 1
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                pred_v, true_v = self.predict(loader=self.data_loader, return_true=True)
                if np.any(np.isnan(pred_v)):
                    warn(
                        "NaN values found in `pred_v`. Replacing with DummyRegressor() values (i.e. mean of training targets)."
                    )
                    pred_v = np.nan_to_num(pred_v)
                mae_v = mean_absolute_error(true_v, pred_v)
                # https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py
                # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
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

    def _add_jitter(self, src, frac, type="normal"):
        """Add a small jitter to the input fractions.

        This improves model robustness and increases stability.

        Parameters
        ----------
        src : torch.tensor
            Tensor containing integers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing the fractions of each element in compound
        type : str, optional
            How to add the jitter. Possible options are "normal" and "uniform". By
            default, "normal"

        Returns
        -------
        frac : torch.tensor
            Tensor containing the fractions of each element in compound with added jitter.
        """
        if type == "normal":
            frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
        elif type == "uniform":
            frac = frac * (1 + (torch.rand_like(frac) - 0.5) * self.fudge)  # uniform
        else:
            raise NotImplementedError(f"{type} not supported as jitter type.")
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        return frac

    def _get_epochs_checkin_stepsize(
        self, epochs: Optional[int], checkin: Optional[int]
    ):
        """Automatically assign epochs, checkin point, and stepsize.

        Parameters
        ----------
        epochs : int, optional
            How many epochs (# of passes through entire dataset). If None, then this is
            automatically assigned based on the dataset size. The number of epochs must be even. By default None
        checkin : int, optional
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None

        Returns
        -------
        epochs : int
            How many epochs (# of passes through entire dataset). If None, then this is
            automatically assigned based on the dataset size. The number of epochs must be even. By default None
        checkin : int
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None
        stepsize : int
            This is equal to `self.epochs_step * len(self.train_loader)`, or in other
            the number of batches that are processed within each set of `epochs_step`.
        """
        assert self.train_loader is not None
        stepsize = self.epochs_step * len(self.train_loader)
        if self.verbose:
            print(
                f"stepping every {stepsize} training passes, cycling lr every {self.epochs_step} epochs"
            )
        if epochs is None:
            # n_iterations = 1e4
            # epochs = max(int(n_iterations / len(self.data_loader)), 40)
            epochs = 300
            if self.verbose:
                print(f"running for {epochs} epochs, unless early stopping occurs")
        if checkin is None:
            checkin = self.epochs_step * 2
            if self.verbose:
                print(
                    f"checkin at {self.epochs_step*2} " f"epochs to match lr scheduler"
                )
        assert isinstance(epochs, int)
        mod = epochs % (self.epochs_step * 2)
        if mod != 0:
            updated_epochs = epochs + (self.epochs_step * 2) - mod
            if self.verbose:
                print(
                    f"{epochs} epochs not divisible by {self.epochs_step * 2} (2*epochs_step), "
                    f"updating epochs to {updated_epochs} for learning"
                )
            epochs = updated_epochs
        return epochs, checkin, stepsize

    def _losscurve_stats(self, epochs, epoch):
        """Compute loss curve statistics.

        Parameters
        ----------
        epochs : int
            How many epochs (# of passes through entire dataset). If None, then this is
            automatically assigned based on the dataset size. The number of epochs must be even. By default None
        epoch : int
            The current epoch.
        """
        pred_t, true_t = self.predict(
            loader=self.train_loader,
            return_true=True,
        )

        if np.any(np.isnan(pred_t)):
            warn(
                "NaN values found in `pred_t`. Replacing with mean of training targets."
            )
            pred_t = np.nan_to_num(pred_t, np.mean(true_t))

        mae_t = mean_absolute_error(true_t, pred_t)
        self.loss_curve["train"].append(mae_t)
        pred_v, true_v = self.predict(loader=self.data_loader, return_true=True)

        if np.any(np.isnan(pred_v)):
            warn(
                "NaN values found in `pred_v`. Replacing with mean of validation targets."
            )
            pred_v = np.nan_to_num(pred_v, np.mean(true_v))

        mae_v = mean_absolute_error(true_v, pred_v)
        self.loss_curve["val"].append(mae_v)
        epoch_str = f"Epoch: {epoch}/{epochs} ---"
        train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
        val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
        if self.classification:
            train_auc = roc_auc_score(true_t, pred_t)
            val_auc = roc_auc_score(true_v, pred_v)
            train_str = f"train auc: {train_auc:0.3f}"
            val_str = f"val auc: {val_auc:0.3f}"
        if self.verbose:
            print(epoch_str, train_str, val_str)

        if self.epoch >= (self.epochs_step * self.swa_start - 1):
            if (self.epoch + 1) % (self.epochs_step * 2) == 0:
                self.xswa.append(self.epoch)
                self.yswa.append(mae_v)

    def _track_stats(self, epochs, checkin, learningcurve, epoch):
        """Track performance statistics of the learning process.

        Parameters
        ----------
        epochs : int
            How many epochs (# of passes through entire dataset). If None, then this is
            automatically assigned based on the dataset size. The number of epochs must be even. By default None
        checkin : int
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None
        learningcurve : bool, optional
            Whether to plot the learning curve, by default True
        epoch : int
            The current epoch.
        """
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

            if learningcurve:
                self._plot_learningcurve(checkin)

    def _plot_learningcurve(self, checkin: int):
        """Plot the learning curve periodically (beginning, checkin, end).

        checkin : int
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None
        """
        if self.learningcurve_fig is None:
            self.learningcurve_fig = plt.figure(figsize=(8, 5))
        else:
            plt.cla()
        xval = np.arange(len(self.loss_curve["val"])) * checkin - 1
        xval[0] = 0
        plt.plot(
            xval,
            self.loss_curve["train"],
            "o-",
            label=r"""$\mathrm{MAE}_\mathrm{train}$""",
        )
        plt.plot(
            xval,
            self.loss_curve["val"],
            "s--",
            label=r"""$\mathrm{MAE}_\mathrm{val}$""",
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
        # plt.show()
        # https://stackoverflow.com/a/56119926/13697228
        assert self.learningcurve_fig is not None
        self.learningcurve_fig.canvas.draw()
        plt.pause(0.01)

    def _plot_losscurve(self, checkin):
        """Plot the loss curve periodically (beginning, checkin, end).

        Parameters
        ----------
        checkin : int
            When to do the checkin step. If None, then automatically assigned as half
            the number of epochs, by default None
        """
        if self.losscurve_fig is None:
            self.losscurve_fig = plt.figure(figsize=(8, 5))
        else:
            plt.cla()
        xval = np.arange(len(self.loss_curve["val"])) * checkin - 1
        xval[0] = 0
        plt.plot(xval, self.loss_curve["train"], "o-", label="train_mae")
        plt.plot(xval, self.loss_curve["val"], "s--", label="val_mae")
        plt.plot(self.xswa, self.yswa, "o", ms=12, mfc="none", label="SWA point")
        plt.ylim(0, 2 * np.mean(self.loss_curve["val"]))
        plt.title(f"{self.model_name}")
        plt.xlabel("epochs")
        plt.ylabel("MAE")
        plt.legend()
        # https://stackoverflow.com/a/56119926/13697228
        assert self.losscurve_fig is not None
        self.losscurve_fig.canvas.draw()
        plt.pause(0.01)

    def _select_criterion(self, criterion: Optional[Union[str, Callable]]):
        """Automatically select a criterion if None was specified.

        Parameters
        ----------
        criterion : Union[str, Callable]
            If a str, then must be one of "RobustL1", "RobustL2", or
            "BCEWithLogitsLoss". If None and classification, then
            `BCEWithLogitsLoss` is used. If None and not `classification`, then
            `RobustL1` is used. If a Callable, then it must follow a similar API to
            e.g. `RobustL1`.
        """
        if criterion is None:
            if self.classification:
                if self.verbose:
                    print("Using BCE loss for classification task")
                self.criterion = BCEWithLogitsLoss
            else:
                self.criterion = RobustL1
        elif type(criterion) is str:
            self.criterion = self.criterion_lookup[criterion]
        else:
            self.criterion = criterion

    def _load_trainval_data(
        self,
        batch_size: int,
        train_data: Union[str, pd.DataFrame],
        val_data: Union[str, pd.DataFrame],
        extra_train_data: pd.DataFrame,
        extra_val_data: pd.DataFrame,
    ):
        """Load both the training and validation data via PyTorch Dataloaders.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to use during training. By default 2 ** 9
        train_data, val_data : Union[str, pd.DataFrame]
            Either a path to the data file or a DataFrame containing at minimum
         and "target" columns for training and validation data, respectively.
        extra_train_data, extra_val_data : pd.DataFrame
            DataFrame containing feature data for columns in `train_data` and `val_data`
         by `extend_features` for training and validation data, respectively.
        """
        self.load_data(
            train_data,
            batch_size=batch_size,
            train=True,
            extra_features=extra_train_data,
        )
        if self.verbose:
            print(
                f"training with batchsize {batch_size} "
                f"(2**{np.log2(batch_size):0.3f})"
            )
        if val_data is not None:
            self.load_data(
                val_data, batch_size=batch_size, extra_features=extra_val_data
            )

        assert_train_str = "Please Load Training Data (self.train_loader)"
        assert_val_str = "Please Load Validation Data (self.data_loader)"
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str

    def _default_batch_size(self, batch_size: Optional[int], data_size: int):
        """Assign a default batch size based on the size of the dataset.

        Parameters
        ----------
        batch_size : int
            The batch size to use during training. If not None, then used as-is. If
            specified, then it is assigned either 2 ** 7 == 128 or 2 ** 12 == 4096
            based on the value of `data_size`.
        data_size : int
            The number of training datapoints.

        Returns
        -------
        batch_size : int
            The batch size to use during training. If not None, then used as-is. If
            specified, then it is assigned either 2 ** 7 == 128 or 2 ** 12 == 4096
            based on the value of `data_size`.
        """
        # Load the train and validation data before fitting the network
        if batch_size is None:
            batch_size = 2 ** round(np.log2(data_size) - 4)
            if batch_size < 2**7:
                batch_size = 2**7
            if batch_size > 2**12:
                batch_size = 2**12
        return batch_size

    def _separate_extended_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        data_dir: Union[str, PathLike],
    ):
        """Extract extra features specified by `extend_features` from data.

        Because `data_dir` can be specified instead of `train_df` and `val_df`, the data
        will be read twice if using a file: once to pull out the extra features here,
        and once when the dataset is loaded into the Dataloader.

        Parameters
        ----------
        train_df, val_df : pd.DataFrame, optional
            Training and validation data with at minimum, "formula" and "target" columns
            and optionally, "extra features" (based on names in `extend_features`). If
            `val_df` is None, then `test_size` determines the amount of training data to
            be split into `val_df`. By default None
        data_dir : str, optional
            The directory from which to load data if loading from a file rather than
            a DataFrame. `data_dir` is only used if both `train_df` and `val_df` are
            None. It is assumed that the files in the data directory will be named
            `train.csv`, `val.csv`, and `test.csv`. By default join(dirname(__file__), "data", "materials_data")

        Returns
        -------
        train_data, val_data : pd.DataFrame
            Training and validation data, respectively. Either the original DataFrame
            or, if `train_df` and `val_df` were both None, the path to the training and
            validation data. If paths are returned, the training and validation data is
            assumed to be located in the following two directories:
            ``join(data_dir, self.mat_prop, "train.csv")``
            ``join(data_dir, self.mat_prop, "val.csv")``
        data_size : int
            The number of training datapoints. While not entirely necessary, this is
            what mainly causes the need to the data twice. It's difficult to know the
            dataset size beforehand without first the data.
        extra_train_data, extra_val_data : pd.DataFrame
            Extra training and validation data, respectively. These are the feature data
            corresponding to the column names in `extend_features`, such as state
            variables (e.g. applied load or temperature).
        """
        if train_df is None and val_df is None:
            use_path = True
        else:
            use_path = False

        if val_df is None:
            # val_df gets used for hyperparameter optimization to improve generalizability
            train_df, val_df = train_test_split(train_df, test_size=self.val_size)

        if use_path:
            # Get the datafiles you will learn from
            assert self.mat_prop is not None
            train_data = join(data_dir, self.mat_prop, "train.csv")
            try:
                val_data = join(data_dir, self.mat_prop, "val.csv")
            except IOError:
                print(
                    "Please ensure you have train (train.csv) and validation data",
                    f'(val.csv) in folder "data/materials_data/{self.mat_prop}"',
                )
            train_df_tmp = pd.read_csv(train_data)
            val_df_tmp = pd.read_csv(val_data)
            data_size = pd.read_csv(train_data).shape[0]
            if self.extend_features is not None:
                extra_train_data = train_df_tmp[self.extend_features]
                extra_val_data = val_df_tmp[self.extend_features]
            else:
                extra_train_data = None
                extra_val_data = None
        else:
            train_data = train_df
            val_data = val_df
            if self.extend_features is not None:
                extra_train_data = train_df[self.extend_features]
                extra_val_data = val_df[self.extend_features]
            else:
                extra_train_data = None
                extra_val_data = None
            assert isinstance(train_data, pd.DataFrame)
            data_size = train_data.shape[0]
        return train_data, val_data, data_size, extra_train_data, extra_val_data

    def save_network(self, model_name: str = None):
        """Save network weights to a ``.pth`` file.

        Parameters
        ----------
        model_name : str, optional
            The name of the `.pth` file. If None, then use `self.model_name`. By default None
        """
        if model_name is None:
            model_name = self.model_name
            os.makedirs(join("models", "trained_models"), exist_ok=True)
            path = join("models", "trained_models", f"{model_name}.pth")
        if self.verbose:
            print(f"Saving network ({model_name}) to {path}")

        assert isinstance(self.model, SubCrab)
        self.network = {
            "weights": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "model_name": model_name,
        }
        torch.save(self.network, path)

    def load_network(self, model_data: Union[str, dict]):
        """Load network weights from a ``.pth`` file.

        Parameters
        ----------
        model_data : Union[str, Any]
            Either the filename of the saved model or the network (see `self.network`)
            as a dictionary of the form:

                {
                "weights": self.model.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "model_name": model_name,
                }
        """
        if type(model_data) is str:
            path = join("models", "trained_models", model_data)
            network = torch.load(path, map_location=self.compute_device)
        else:
            network = model_data
        assert isinstance(self.model, SubCrab)
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(network["weights"])
        self.scaler.load_state_dict(network["scaler_state"])
        self.model_name = network["model_name"]

    def load_model(
        self,
        model: Union[str, SubCrab],
        data: Union[str, pd.DataFrame],
        classification: bool = False,
        verbose: bool = True,
    ):
        """Load a _CrabNet model.

        Parameters
        ----------
        model : Union[str, _CrabNet]
            The CrabNet model to load or the filename of the saved network.
        data : Union[str, pd.DataFrame]
            The data to load, which can be the folder in which the ``.csv`` file resides
            or a pandas DataFrame.
        classification : bool, optional
            Whether to perform classification. If False, then assume regression. By
            default, False
        verbose : bool, optional
            Whether model information and progress should be printed, by default True
        """
        # Load up a saved network.
        if type(model) is str:
            self.load_network(model)

        # Check if classifcation task
        if classification:
            self.classification = True

        # data is reloaded to self.data_loader
        self.load_data(data, batch_size=2**9, train=False)


# %%
if __name__ == "__main__":
    pass
