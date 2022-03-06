import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=False)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def get_model(mat_prop, i, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}{i}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'data/matbench_cv/{mat_prop}/train{i}.csv'
    val_data = f'data/matbench_cv/{mat_prop}/val{i}.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_matbench__predictions'
    # save_path = 'publication_predictions/onehot_matbench__predictions'
    # save_path = 'publication_predictions/random_200_matbench__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(mat_prop, i, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}{i}', verbose=verbose)
    model.load_network(f'{mat_prop}{i}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'data/matbench_cv/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(mat_prop, i, classification, file_name, verbose=True):
    model = load_model(mat_prop, i, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'\n{mat_prop}{i} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'\n{mat_prop}{i} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(f"{i}.csv", "")}_output_cv{i}.csv'
    to_csv(output, fname)
    return model, mae


# %%
if __name__ == '__main__':
    # Get data to benchmark on
    data_dir = 'data/matbench_cv'
    mat_props = os.listdir(data_dir)
    classification_list = []
    print(f'training: {mat_props}')
    for mat_prop in mat_props:
        classification = False
        if mat_prop in classification_list:
            classification = True
        # matbench provides 5 dataset train/val splits
        n_splits = 5
        maes = []
        for i in range(n_splits):
            print(f'property: {mat_prop}, cv {i}')
            model = get_model(mat_prop, i, classification, verbose=True)
            print('=====================================================')
            print('calculating test mae')
            model_test, t_mae = save_results(mat_prop, i, classification,
                                             f'test{i}.csv', verbose=False)
            print('calculating val mae')
            model_val, v_mae = save_results(mat_prop, i, classification,
                                            f'val{i}.csv', verbose=False)
            maes.append(t_mae)
        print(f'Average test mae: {np.mean(maes)}')
        print('=====================================================')
