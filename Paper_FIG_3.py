import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize

compute_device = get_compute_device()
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


def model(mat_prop, classification_list, simple=False):
    # Get the TorchedCrabNet architecture loaded
    model = Model(
        CrabNet(compute_device=compute_device).to(compute_device),
        model_name=f"{mat_prop}",
    )
    if True:
        model.load_network(f"{mat_prop}.pth")
        model.model_name = f"{mat_prop}"

    if mat_prop in classification_list:
        model.classification = True

    dataset = rf"{data_dir}\{mat_prop}\train.csv"
    model.load_data(dataset, batch_size=2 ** 7)  # data is reloaded to model.data_loader

    model.model.eval()
    model.model.avg = False

    simple_tracker = {i: [] for i in range(119)}
    element_tracker = {i: [] for i in range(119)}
    composition_tracker = {}

    with torch.no_grad():
        for i, data in enumerate(tqdm(model.data_loader)):
            X, y, formula = data
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            src = src.to(compute_device, dtype=torch.long, non_blocking=True)
            frac = frac.to(compute_device, dtype=data_type, non_blocking=True)
            y = y.to(compute_device, dtype=data_type, non_blocking=True)
            output = model.model.forward(src, frac)
            mask = (src == 0).unsqueeze(-1).repeat(1, 1, 1)
            prediction, uncertainty, prob = output.chunk(3, dim=-1)
            prediction = prediction * torch.sigmoid(prob)
            uncertainty = torch.exp(uncertainty) * model.scaler.std
            prediction = model.scaler.unscale(prediction)
            prediction = prediction * ~mask
            uncertainty = uncertainty * ~mask
            if model.classification:
                prediction = torch.sigmoid(prediction)
            for i in range(src.shape[0]):
                if any(prediction[i].cpu().numpy().ravel() < 0):
                    composition_tracker[formula[i]] = [
                        src[i].cpu().numpy(),
                        frac[i].cpu().numpy(),
                        y[i].cpu().numpy(),
                        prediction[i].cpu().numpy(),
                        uncertainty[i].cpu().numpy(),
                    ]
                for j in range(src.shape[1]):
                    element_tracker[int(src[i][j])].append(float(prediction[i][j]))
                    simple_tracker[int(src[i][j])].append(float(y[i]))

    def elem_view(element_tracker, plot=True):
        property_tracker = {}
        x_max = max([y[1] for y in model.data_loader.dataset])
        x_min = min([y[1] for y in model.data_loader.dataset])
        x_range = x_max - x_min
        x_min_buffer = 0.1 * x_range
        x_max_buffer = 0.1 * x_range
        for key in element_tracker.keys():
            data = element_tracker[key]
            if len(data) > 10:
                sum_prop = sum(data)
                mean_prop = sum_prop / len(data)
                prop = mean_prop
                property_tracker[all_symbols[key]] = prop
                if plot:
                    plt.figure(figsize=(4, 4))
                    hist_kws = {
                        "edgecolor": "k",
                        "linewidth": 2,
                        "alpha": 1,
                        "facecolor": "#A1D884",
                    }
                    ax = sns.distplot(
                        data,
                        label=f"{all_symbols[key]}, n={len(data)}",
                        kde=False,
                        bins=np.arange(0, 500, 25),
                        hist_kws=hist_kws,
                        kde_kws={"color": "k", "linewidth": 2},
                    )

                    ax.axes.yaxis.set_visible(False)
                    plt.legend()
                    plt.xlim(x_min - x_min_buffer, x_max + x_max_buffer)
                    plt.xlabel("Bulk Modulus Contribution (GPa)")
                    plt.tick_params(axis="both", which="both", direction="in")

                    save_dir = f"figures/contributions/{mat_prop}/"
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        f"{save_dir}{all_symbols[key]}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.show()
        return property_tracker

    if simple:
        property_tracker = elem_view(simple_tracker, plot=True)
    else:
        property_tracker = elem_view(element_tracker, plot=True)

    return property_tracker


def save_results(output, save_name):
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ["composition", "target", "pred-0", "uncertainty"]
    save_path = "new_results"


def save_test_results(mat_prop, classification_list):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device))
    model.load_network(f"{mat_prop}.pth")
    if mat_prop in classification_list:
        model.classification = True
    # Load the data you want to predict with
    test_data = rf"data\benchmark_data\{mat_prop}\test.csv"
    model.load_data(test_data)  # data is reloaded to model.data_loader
    output = model.predict(model.data_loader)  # predict the data saved here
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f"\n{mat_prop} ROC AUC: {auc:0.3f}")
    else:
        print(f"\n{mat_prop} mae: {abs(output[0] - output[1]).mean():0.3f}")
    # save your predictions to a csv
    save_results(output, f"{mat_prop}_output.csv")


# %%
def plot(mat_prop, property_tracker):
    ptable = pd.read_csv("data/element_properties/ptable.csv")
    ptable.index = ptable["symbol"].values
    elem_tracker = ptable["count"]
    n_row = ptable["row"].max()
    n_column = ptable["column"].max()

    elem_tracker = elem_tracker + pd.Series(property_tracker).drop("None", axis=0)

    # log_scale = True
    log_scale = False

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable["row"]
    columns = ptable["column"]
    symbols = ptable["symbol"]
    rw = 0.9  # rectangle width (rw)
    rh = rw  # rectangle height (rh)
    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable["row"].max() - row
        cmap = cm.YlGn
        count_min = 0
        count_max = elem_tracker.max() + 50
        norm = Normalize(vmin=count_min, vmax=count_max)
        count = elem_tracker[symbol]
        if log_scale:
            norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
            if count != 0:
                count = np.log(count)
        color = cmap(norm(count))
        if np.isnan(count):
            color = "silver"
        if row < 3:
            row += 0.5
        # element box
        rect = patches.Rectangle(
            (column, row),
            rw,
            rh,
            linewidth=1.5,
            edgecolor="gray",
            facecolor=color,
            alpha=1,
        )
        # plot element text
        plt.text(
            column + rw / 2,
            row + rw / 2,
            symbol,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=22,
            fontweight="semibold",
            color="k",
        )

        ax.add_patch(rect)

    granularity = 20
    for i in range(granularity):
        value = (1 - i / (granularity - 1)) * count_min + (
            i / (granularity - 1)
        ) * count_max
        if log_scale:
            if value != 0:
                value = np.log(value)
        color = cmap(norm(value))
        length = 9
        x_offset = 3.5
        y_offset = 7.8
        x_loc = i / (granularity) * length + x_offset
        width = length / granularity
        height = 0.35
        rect = patches.Rectangle(
            (x_loc, y_offset),
            width,
            height,
            linewidth=1.5,
            edgecolor="gray",
            facecolor=color,
            alpha=1,
        )

        if i in [0, 4, 9, 14, 19]:
            text = f"{value:0.0f}"
            if log_scale:
                text = f"{np.exp(value):0.1e}".replace("+", "")
            plt.text(
                x_loc + width / 2,
                y_offset - 0.4,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="semibold",
                fontsize=20,
                color="k",
            )

        ax.add_patch(rect)

    name = "Bulk Modulus"
    legend_title = f"Average {name} Contribution"
    plt.text(
        x_offset + length / 2,
        y_offset + 0.7,
        f"log({legend_title})" if log_scale else legend_title,
        horizontalalignment="center",
        verticalalignment="center",
        fontweight="semibold",
        fontsize=20,
        color="k",
    )

    ax.set_ylim(-0.15, n_row + 0.1)
    ax.set_xlim(0.85, n_column + 1.1)

    # fig.patch.set_visible(False)
    ax.axis("off")

    plt.draw()
    save_dir = "figures/contributions"
    # save_dir = None
    if save_dir is not None:
        name = mat_prop
        fig_name = f"{save_dir}/{mat_prop}/ptable.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches="tight", dpi=300)

    plt.pause(0.001)
    plt.close()


# %%
data_dir = "data/benchmark_data"
matbench_props = os.listdir(data_dir)[0:1]

classification_list = []

mat_prop = "aflow__ael_bulk_modulus_vrh"
property_tracker = model(mat_prop, classification_list)
plot(mat_prop, property_tracker)
