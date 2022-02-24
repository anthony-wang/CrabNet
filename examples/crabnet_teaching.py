"""Simplified teaching example for CrabNet. Use for real problems is discouraged."""
from os import PathLike
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from crabnet.kingcrab import ResidualNetwork, SubCrab

from crabnet.data.materials_data import elasticity
from crabnet.utils.data import get_data

from crabnet.utils.utils import (
    BCEWithLogitsLoss,
    DummyScaler,
    EDM_CsvLoader,
    Lamb,
    Lookahead,
    RobustL1,
    Scaler,
)

dummy = True

if dummy:
    compute_device = torch.device("cpu")
else:
    compute_device = torch.device("cuda")

train_df, val_df = get_data(elasticity, dummy=dummy)

# %% parameters
model_name: str = "elasticity"
n_elements: Union[str, int] = "infer"
classification: bool = False
batch_size = 128
epochs: Optional[int] = 4
epochs_step: int = 1
checkin: Optional[int] = 2
out_dims: int = 3
d_model: int = 512
heads: int = 4
elem_prop: str = "mat2vec"
bias = False
dropout: float = 0.1
out_hidden: List[int] = [1024, 512, 256, 128]
lr: float = 1e-3
betas: Tuple[float, float] = (0.9, 0.999)
eps: float = 1e-6
weight_decay: float = 0
adam: bool = False
alpha: float = 0.5
k: int = 6
base_lr: float = 1e-4
max_lr: float = 6e-3
random_state: Optional[int] = 42
mat_prop: Optional[Union[str, PathLike]] = "elasticity"

criterion: Callable
if classification:
    criterion = BCEWithLogitsLoss
else:
    criterion = RobustL1

data_type_torch = torch.float32

# %% load training and validation data into PyTorch Dataloader
# training
train = True
inference = not train
data_loaders = EDM_CsvLoader(
    data=train_df,
    batch_size=batch_size,
    n_elements=n_elements,
    inference=inference,
    elem_prop=elem_prop,
)

train_loader = data_loaders.get_data_loaders(inference=inference)
y = train_loader.dataset.data[1]
if classification:
    scaler: Union[Scaler, DummyScaler] = DummyScaler(y)
else:
    scaler = Scaler(y)

# validation
train = False
inference = not train
data_loaders = EDM_CsvLoader(
    data=val_df,
    batch_size=batch_size,
    n_elements=n_elements,
    inference=inference,
    elem_prop=elem_prop,
)

val_loader = data_loaders.get_data_loaders(inference=inference)
y = val_loader.dataset.data[1]

model = SubCrab(
    compute_device=compute_device,
    out_dims=out_dims,
    d_model=d_model,
    heads=heads,
    dropout=dropout,
).to(compute_device)

step_count = 0

base_optim = Lamb(params=model.parameters())
optimizer = Lookahead(base_optimizer=base_optim, alpha=alpha, k=k)

# removed: stochastic weight averaging and learning rate scheduler

for epoch in range(epochs):
    # %% training
    for data in train_loader:
        # separate into src and frac
        X, y, _, _ = data
        y = scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)

        # send to PyTorch device
        src = src.to(compute_device, dtype=torch.long, non_blocking=True)
        frac = frac.to(compute_device, dtype=data_type_torch, non_blocking=True)
        y = y.to(compute_device, dtype=data_type_torch, non_blocking=True)

        # train
        output = model.forward(src, frac)
        prediction, uncertainty = output.chunk(2, dim=-1)
        loss = criterion(prediction.view(-1), uncertainty.view(-1), y.view(-1))

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# %% predict

len_dataset = len(val_loader.dataset)
n_atoms = int(len(val_loader.dataset[0][0]) / 2)
act = np.zeros(len_dataset)
pred = np.zeros(len_dataset)
uncert = np.zeros(len_dataset)
formulae = np.empty(len_dataset, dtype=list)
atoms = np.empty((len_dataset, n_atoms))
fractions = np.empty((len_dataset, n_atoms))

model.eval()

with torch.no_grad():
    for i, valbatch_df in enumerate(val_loader):
        # extract data
        X, y, formula, extra_features = valbatch_df
        src, frac = X.squeeze(-1).chunk(2, dim=1)

        # send to device
        src = src.to(compute_device, dtype=torch.long, non_blocking=True)
        frac = frac.to(compute_device, dtype=data_type_torch, non_blocking=True)
        y = y.to(compute_device, dtype=data_type_torch, non_blocking=True)
        extra_features = extra_features.to(
            compute_device, dtype=data_type_torch, non_blocking=True
        )

        # predict
        output = model.forward(src, frac, extra_features=extra_features)
        prediction, uncertainty = output.chunk(2, dim=-1)
        uncertainty = torch.exp(uncertainty) * scaler.std
        prediction = scaler.unscale(prediction)
        if classification:
            prediction = torch.sigmoid(prediction)

        data_loc = slice(i * batch_size, i * batch_size + len(y), 1)

        atoms[data_loc, :] = src.cpu().numpy()
        fractions[data_loc, :] = frac.cpu().numpy()
        act[data_loc] = y.view(-1).cpu().numpy()
        pred[data_loc] = prediction.view(-1).cpu().detach().numpy()
        uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy()
        formulae[data_loc] = formula

print(act)
print(pred)
print(uncert)

dummy_mae = mean_absolute_error(act, np.mean(train_df["target"]) * np.ones_like(act))
mae = mean_absolute_error(act, pred)
print(f"Dummy MAE: {dummy_mae :.3f}")
print(f"MAE: {mae :.3f}")
