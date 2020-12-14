import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = rf'{data_dir}\{mat_prop}\train.csv'
    try:
        val_data = rf'{data_dir}\{mat_prop}\val.csv'
    except:
        print('Please ensure you have train (train.csv) and validation data',
               f'(val.csv) in folder "data\materials_data\{mat_prop}"')

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=40, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'model_predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(f'{mat_prop}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = rf'{data_dir}\{mat_prop}\{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'{mat_prop} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'{mat_prop} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)
    return model, mae


# %%
if __name__ == '__main__':
    # Choose the directory where your data is stored
    data_dir = 'data/materials_data'
    # Choose the folder with your materials properties
    mat_prop = 'example_materials_property'
    # Choose if you data is a regression or binary classification
    classification = False
    # train = False
    train = True

    # Train your model using the "get_model" function
    if train:
        print(f'Property "{mat_prop}" selected for training')
        model = get_model(data_dir, mat_prop, classification, verbose=True)

    cutter = '====================================================='
    first = " "*((len(cutter)-len(mat_prop))//2) + " "*int((len(mat_prop)+1)%2)
    last = " "*((len(cutter)-len(mat_prop))//2)
    print('=====================================================')
    print(f'{first}{mat_prop}{last}')
    print('=====================================================')
    print('calculating train mae')
    model_train, mae_train = save_results(data_dir, mat_prop, classification,
                                          'train.csv', verbose=False)
    print('-----------------------------------------------------')
    print('calculating val mae')
    model_val, mae_valn = save_results(data_dir, mat_prop, classification,
                                       'val.csv', verbose=False)
    print('-----------------------------------------------------')
    print('calculating test mae')
    model_test, mae_test = save_results(data_dir, mat_prop, classification,
                                        'test.csv', verbose=False)
    print('=====================================================')

