import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib.pyplot as plt
from time import time
from utils.get_compute_device import get_compute_device
from utils.utils import EDM_CsvLoader

from crabnet.kingcrab import CrabNet
from crabnet.model import Model

compute_device = get_compute_device(prefer_last=True)
print(f"Running on compute device: {compute_device}")


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type = torch.float


# %%
# fmt: off
all_symbols = ['None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
               'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
               'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
               'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
               'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
               'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
               'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
               'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
               'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
# fmt: on

classification = False
batch_size = None
transfer = None
verbose = True
# Get the TorchedCrabNet architecture loaded
mat_prop = "steels_yield"
file_name = rf"data\matbench_cv\{mat_prop}\val0.csv"

model = Model(
    CrabNet(compute_device=compute_device).to(compute_device),
    model_name=f"{mat_prop}",
    verbose=verbose,
)

batch_size = 1
edm_loader = EDM_CsvLoader(csv_data=file_name, batch_size=batch_size)
data_loader = edm_loader.get_data_loaders(inference=True)


# %%
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mat2vec = True
        if self.mat2vec:
            mat2vec = "data/element_properties/mat2vec.csv"
            # mat2vec = 'data/element_properties/onehot.csv'
            # mat2vec = 'data/element_properties/oliynyk.csv'
            cbfv = pd.read_csv(mat2vec, index_col=0).values
            feat_size = cbfv.shape[-1]
            self.fc_mat2vec = nn.Linear(feat_size, d_model)
            self.fc_mat2vec = self.fc_mat2vec.to(compute_device, dtype=torch.float)
            zeros = np.zeros((1, feat_size))
            cat_array = np.concatenate([zeros, cbfv])
            self.cbfv = torch.tensor(cat_array).to(compute_device, dtype=torch.float)

    def forward(self, src):
        mat2vec = self.cbfv[src]
        # x_emb = mat2vec
        x_emb = self.fc_mat2vec(mat2vec)
        return x_emb


class FractionalEncoder(nn.Module):
    """
    Encoding element fractions using encoding sytle a la Vaswani 2017.
    Unsqueezed the fractional encoding a bit so that we are using more space
    in the d_model dimension to encode fractional information.
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
        x[x < 1 / self.resolution] = 1 / self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
ti = time()
for i, data in enumerate(data_loader):
    X, y, formula = data
    src, frac = X.squeeze(-1).chunk(2, dim=1)
    src = src.to(compute_device, dtype=torch.long, non_blocking=True)
    frac = frac.to(compute_device, dtype=data_type, non_blocking=True)
    y = y.to(compute_device, dtype=data_type, non_blocking=True)

X, y, formula = data_loader.dataset[23]
src, frac = X.chunk(2, dim=0)
src = src.to(compute_device, dtype=torch.long, non_blocking=True)
frac = frac.to(compute_device, dtype=data_type, non_blocking=True)
y = y.to(compute_device, dtype=data_type, non_blocking=True)
print(i, src)
print(i, frac)
print(src.shape)
print(src.shape)

src[:, :] = 0
src[0, :] = 8
src[1, :] = 13

frac[:, :] = 0
frac[0, :] = 0.6
frac[1, :] = 0.4

vocab_size = 118
d_model = 512

embedder = model.model.encoder.embed
pe = model.model.encoder.pe
ple = model.model.encoder.ple


x = embedder(src)
x = x.masked_fill(src.unsqueeze(1).repeat(1, 1, x.shape[-1]) == 0, 0)

pe_tensor = torch.zeros_like(x)
ple_tensor = torch.zeros_like(x)

pe_tensor[:, :, : d_model // 2] = pe(frac)
ple_tensor[:, :, d_model // 2 :] = ple(frac)

x_cpu = x.cpu().detach().numpy()
pe_tensor_cpu = pe(frac).cpu().detach().numpy()
ple_tensor_cpu = ple(frac).cpu().detach().numpy()

print(x_cpu.shape)
print(pe_tensor_cpu.shape)
print(ple_tensor_cpu.shape)

x_cpu = x_cpu.squeeze(1)
pe_tensor_cpu = pe_tensor_cpu.squeeze(1)
ple_tensor_cpu = ple_tensor_cpu.squeeze(1)

print(x_cpu.shape)
print(pe_tensor_cpu.shape)
print(ple_tensor_cpu.shape)
frac = frac.squeeze(1)
src = src.squeeze(1)


# %%
# colors = ['#7fc97f', '#beaed4', '#fdc086', 'r', 'g', 'steelblue']
colors = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
nrows = len(src)


fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, figsize=(8, 8))
for i, f in enumerate(src):
    element = all_symbols[src[i]]
    axes[i].plot(x_cpu[i], label=f"{element}", color=colors[i])
    axes[i].legend(loc=1, framealpha=0.4, handlelength=0)
    axes[i].set_yticklabels([])
    axes[i].set_facecolor("#202020")
plt.tick_params(which="major", right=True, top=True, direction="in", length=6)
plt.tick_params(which="minor", right=True, top=True, direction="in", length=4)
plt.show()


fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, figsize=(8, 8))
for i, f in enumerate(frac):
    axes[i].plot(pe_tensor_cpu[i], label=f"{f:0.3f}", color=colors[i])
    axes[i].legend(loc=1, framealpha=0.4, handlelength=0)
    axes[i].set_yticklabels([])
    axes[i].set_facecolor("#202020")
plt.tick_params(which="major", right=True, top=True, direction="in", length=6)
plt.tick_params(which="minor", right=True, top=True, direction="in", length=4)
plt.show()


fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, figsize=(8, 8))
for i, f in enumerate(frac):
    axes[i].plot(ple_tensor_cpu[i], label=f"{f:0.0e}", color=colors[i])
    axes[i].legend(loc=1, framealpha=0.4, handlelength=0)
    axes[i].set_yticklabels([])
    axes[i].set_facecolor("#202020")
plt.tick_params(which="major", right=True, top=True, direction="in", length=6)
plt.tick_params(which="minor", right=True, top=True, direction="in", length=4)
plt.show()


fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, figsize=(8, 8))
for i, f in enumerate(frac):
    element = all_symbols[src[i]]
    ebfv = (
        x_cpu[i] * 2
        + pe_tensor[i].cpu().detach().numpy()
        + ple_tensor[i].cpu().detach().numpy()
    )
    ebfv = ebfv.ravel()
    if f == 0:
        ebfv = [0] * len(ebfv)
    axes[i].plot(ebfv, label=f"${element}_{{{f:0.1f}}}$", color=colors[i])
    axes[i].legend(loc=4, framealpha=0.6, handlelength=0)
    axes[i].set_yticklabels([])
    axes[i].set_facecolor("#202020")
plt.tick_params(which="major", right=True, top=True, direction="in", length=6)
plt.tick_params(which="minor", right=True, top=True, direction="in", length=4)
plt.show()
