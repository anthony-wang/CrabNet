import numpy as np

import torch

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
from utils.composition import _element_composition

import matplotlib.pyplot as plt

from collections import defaultdict

compute_device = get_compute_device(prefer_last=False)
data_type_torch = torch.float32


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

color = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

classification_list = []

num = ""
mat_prop = "aflow__Egap"

# Get the TorchedCrabNet architecture loaded
model = Model(CrabNet().to(compute_device), model_name=f"{mat_prop}")
if True:
    model.load_network(f"{mat_prop}{num}.pth")
    model.model_name = f"{mat_prop}{num}"

if mat_prop in classification_list:
    model.classification = True

mat_prop = "aflow__Egap"
test_data = rf"data\benchmark_data\{mat_prop}\train.csv"
# test_data = rf'data\matbench_cv\{mat_prop}\train{num}.csv'

model.load_data(test_data, batch_size=2 ** 0)  # data is reloaded to model.data_loader

len_dataset = len(model.data_loader.dataset)
n_atoms = int(len(model.data_loader.dataset[0][0]) / 2)
act = np.zeros(len_dataset)
pred = np.zeros(len_dataset)
uncert = np.zeros(len_dataset)
formulae = np.empty(len_dataset, dtype=list)
atoms = np.empty((len_dataset, n_atoms))
fractions = np.empty((len_dataset, n_atoms))
model.model.eval()
model.model.avg = False

simple_tracker = {i: [] for i in range(119)}
variance_tracker = {i: [] for i in range(119)}
element_tracker = {i: [] for i in range(119)}
composition_tracker = {}

alls_dict = defaultdict(list)
binaries_dict = defaultdict(list)
for data in model.data_loader.dataset:
    keys = _element_composition(data[2]).keys()
    if len(keys) == 2:
        binaries_dict[tuple(sorted(keys))].append(data)
    else:
        alls_dict[tuple(sorted(keys))].append(data)


# %%
max_len = 1

for key, systems in binaries_dict.items():
    if "Si" not in key:
        continue
    if "O" not in key:
        continue
    print(key)
    if len(systems) < max_len:
        continue
    og_points = []
    with torch.no_grad():
        for data in systems:
            X, y, formula = data
            print(formula)
            src, frac = X.unsqueeze(0).squeeze(-1).chunk(2, dim=1)
            src = src.to(compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(compute_device, dtype=data_type_torch, non_blocking=True)
            y = y.to(compute_device, dtype=data_type_torch, non_blocking=True)
            output = model.model.forward(src, frac)
            mask = (src == 0).unsqueeze(-1).repeat(1, 1, 3)
            mask = (src == 0).unsqueeze(-1).repeat(1, 1, 1)
            output = output.masked_fill(mask, 0)
            prob = output[:, :, -1:]
            output = output[:, :, :2]
            probability = torch.ones_like(output)
            probability[:, :, :1] = torch.sigmoid(prob)
            output = output * probability

            prediction, uncertainty = output.chunk(2, dim=-1)
            uncertainty = torch.exp(uncertainty) * model.scaler.std
            prediction = model.scaler.unscale(prediction)
            prediction = prediction * ~mask
            uncertainty = uncertainty * ~mask
            slist = src.cpu().numpy().ravel().tolist()
            flist = frac.cpu().numpy().ravel().tolist()
            sf_dict = {src: frac for src, frac in zip(slist, flist)}
            elem_frac = sf_dict[sorted(slist)[-1]]
            og_points.append([float(elem_frac), float(y), formula])

        frac_tracker = []
        pred_tracker = []
        uncertainty_tracker = []
        elem1_tracker = []
        elem2_tracker = []
        res = 50
        X, y, formula = data
        X_data = [(src, frac) for src, frac in zip(slist, flist)]
        X_data = sorted(X_data, reverse=True)
        slist = [x[0] for x in X_data]
        flist = [x[1] for x in X_data]
        X = torch.tensor(slist + flist).unsqueeze(1)

        first = True
        for i in range(0, res - 1):
            src, frac = X.unsqueeze(0).squeeze(-1).chunk(2, dim=1)
            frac = frac.clone()
            src = src.clone()

            frac[:, 0:1] = 1 / res + (i / res)
            frac[:, 1:2] = 1 - frac[:, 0:1]
            src = src.to(compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(compute_device, dtype=data_type_torch, non_blocking=True)
            y = y.to(compute_device, dtype=data_type_torch, non_blocking=True)

            output = model.model.forward(src, frac)
            mask = (src == 0).unsqueeze(-1).repeat(1, 1, 1)
            output = output.masked_fill(mask, 0)
            prob = output[:, :, -1:]
            output = output[:, :, :2]
            probability = torch.ones_like(output)
            probability[:, :, :1] = torch.sigmoid(prob)
            output = output * probability

            prediction, uncertainty = output.chunk(2, dim=-1)
            prediction = model.scaler.unscale(prediction)
            prediction = prediction * ~mask
            uncertainty = uncertainty * ~mask
            uncertainty_mean = (uncertainty * ~mask).sum() / (~mask).sum()
            uncertainty = torch.exp(uncertainty_mean) * model.scaler.std

            min_val = prediction[:, :, :][prediction != 0].min().cpu().detach().numpy()
            max_val = prediction[:, :, :][prediction != 0].max().cpu().detach().numpy()
            range_val = max_val - min_val
            pred_elem1 = prediction[:, 0, :].cpu().detach().numpy().ravel()
            pred_elem2 = prediction[:, 1, :].cpu().detach().numpy().ravel()
            act_val = y.detach().cpu().numpy().ravel()
            uncertainty_tracker.append(uncertainty.cpu().detach().numpy())
            frac_tracker.append(frac[0, :1].cpu().detach().numpy())
            elem1_tracker.append(pred_elem1)
            elem2_tracker.append(pred_elem2)
            pred_tracker.append(
                ((prediction * ~mask).sum() / (~mask).sum()).cpu().detach().numpy()
            )
            elem1_name = all_symbols[int(src[0, 0].cpu().detach().numpy())]
            elem2_name = all_symbols[int(src[0, 1].cpu().detach().numpy())]

        frac_tracker = np.array(frac_tracker).ravel()
        pred_tracker = np.array(pred_tracker).ravel()
        uncertainty_tracker = np.array(uncertainty_tracker).ravel()
        plt.figure(figsize=(7, 5))
        plt.fill_between(
            frac_tracker,
            pred_tracker - uncertainty_tracker,
            pred_tracker + uncertainty_tracker,
            alpha=0.7,
            color="silver",
        )
        plt.plot(
            frac_tracker,
            elem1_tracker,
            "-.",
            label=f"{elem1_name} contribution",
            color=color[0],
            linewidth=2,
        )
        plt.plot(
            frac_tracker,
            elem2_tracker,
            "--",
            label=f"{elem2_name} contribution",
            color=color[1],
            linewidth=2,
        )
        plt.plot(
            frac_tracker,
            pred_tracker,
            "-",
            color="gray",
            label="Prediction",
            linewidth=2,
        )
        plt.xlim(0, 1)
        plt.ylabel("$Si_xO_{1-x}$ Band Gap (eV)")
        plt.xlabel("$x$")

        plt.legend(
            ncol=1, handlelength=2, labelspacing=0.08, columnspacing=1, framealpha=0.3
        )
        plt.savefig(
            f"figures/Figure4_binary_plot_{elem1_name}-{elem2_name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()
