"""Simplified teaching example for CrabNet. Use for real problems is discouraged."""
from pprint import pprint

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from examples.subcrab_teaching import SubCrab

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

train_df, val_df = get_data(elasticity, dummy=True)

compute_device = torch.device("cuda")
batch_size = 32
classification = False

if classification:
    criterion = BCEWithLogitsLoss
else:
    criterion = RobustL1

# %% load training and validation data into PyTorch Dataloader
data_loaders = EDM_CsvLoader(data=train_df, batch_size=batch_size)
train_loader = data_loaders.get_data_loaders()

y = train_loader.dataset.data[1]
if classification:
    scaler = DummyScaler(y)
else:
    scaler = Scaler(y)

data_loaders = EDM_CsvLoader(data=val_df, inference=True, batch_size=batch_size)
val_loader = data_loaders.get_data_loaders(inference=True)

# %% model setup
model = SubCrab().to(compute_device)

base_optim = Lamb(params=model.parameters())
optimizer = Lookahead(base_optimizer=base_optim)

data_type_torch = torch.float32
epochs = 10
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

act = np.zeros(len_dataset)
pred = np.zeros(len_dataset)
uncert = np.zeros(len_dataset)

# set the model to evaluation mode (as opposed to training mode)
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

        # predict
        output = model.forward(src, frac)
        prediction, uncertainty = output.chunk(2, dim=-1)
        uncertainty = torch.exp(uncertainty) * scaler.std
        prediction = scaler.unscale(prediction)
        if classification:
            prediction = torch.sigmoid(prediction)

        # splice batch results into main data
        data_loc = slice(i * batch_size, i * batch_size + len(y), 1)
        act[data_loc] = y.view(-1).cpu().numpy()
        pred[data_loc] = prediction.view(-1).cpu().detach().numpy()
        uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy()

pprint(pred)
pprint(uncert)

dummy_mae = mean_absolute_error(act, np.mean(train_df["target"]) * np.ones_like(act))
print(f"Dummy MAE: {dummy_mae :.3f}")
mae = mean_absolute_error(act, pred)
print(f"MAE: {mae :.3f}")
