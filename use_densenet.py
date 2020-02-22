import os

import warnings
warnings.filterwarnings('ignore',
                        message='PyTorch is not compiled with NCCL support')

import torch

import matplotlib.pyplot as plt

from datetime import datetime

from models.data import EDM_CsvLoader
from models.neuralnetwrapper import NeuralNetWrapper

from utils.utils import CONSTANTS
from utils.get_compute_device import get_compute_device

plt.rcParams.update({'font.size': 16})
cons = CONSTANTS()


# %%
def list_saved_models():
    """
    Print previously trained model names.
    """
    weight_names = os.listdir('data/user_properties/trained_weights')
    model_names = [name.split('CUSTOM_')[-1].split('-checkpoint')[0] for
                   name in weight_names]
    print(f'current saved models: {model_names}')

def load_densenet(mat_prop, batch_size):
    """
    This function will create a densenet instance. This does not need to be
    edited.

    Parameters
    ----------
    mat_prop : str
        Name used for saving model. Names with the string "CUSTOM" in them
        will have their training statistics saved and will have their
        trained model weights saved to data/user_properties/traine_weights
    batch_size : int
        Batch size used for training and inference.

    Returns
    -------
    model : NeuralNetWrapper
        Wrapper object that contains pytorch model as well as the fit,
        predict, and weight loading methods.

    """

    # define the model type, element properties, and input dims
    # These should not be changed for the base "densenet" impelementation
    model_type = 'DenseNet'
    elem_prop = 'element'
    edm = (elem_prop == 'element')
    input_dims = 119
    seed = None

    # values only used if model_type = 'DenseNet'
    hidden_dims = [256, 128, 64]
    output_dims = 1

    # initialize the model using defined parameters
    start_datetime_load_densenet = (datetime.now()
                                    .strftime('%Y-%m-%d-%H%M%S.%f'))
    out_dir = f'data/score_summaries/NN/Run_{start_datetime_load_densenet}/'
    save_network_info = False

    if save_network_info:
        os.makedirs(out_dir, exist_ok=True)
    model = NeuralNetWrapper(model_type,
                             elem_prop,
                             mat_prop,
                             input_dims,
                             hidden_dims,
                             output_dims,
                             out_dir,
                             edm=edm,
                             batch_size=batch_size,
                             random_seed=seed,
                             save_network_info=save_network_info)

    #return a "skeleton" of the model. These weights are untrained
    return model

def train_densenet(model_name, csv_train, csv_val=None, val_frac=0.25):
    """
    Function to train densenet. This function allows a user to easily train
    densenet by only supplying training data a model name. You can update
    epoch and batch size if desired.

    Parameters
    ----------
    model_name : str
        Custom name you would like to use to refer to your model. This will
        be used when you want to generate predictions.
    csv_train : str
        The name of a .csv file located in "data/user_properties". This is
        the data you will train from. This file needs to have two columns:
        "formula" and "target".
    csv_val : str, optional
        The name of a .csv file located in "data/user_properties". This is
        the data you will validate on. This file needs to have two columns:
        "formula" and "target".
    val_frac : float, optional
        The fractional split for train-val separation during training. This
        value is only used if csv_train is not supplied by the user.
        The default is 0.25.

    Returns
    -------
    None.

    """

    pretrained_models = cons.mps
    if model_name in pretrained_models:
        warn_msg = f'model_name "{model_name}" reserved for pretrained model'
        warnings.warn(warn_msg, UserWarning)
        return None

    mat_props_root = r'data/user_properties/'
    csv_train = f'{mat_props_root}{csv_train}'
    batch_size = 2**10
    epochs = 501
    data_loaders = EDM_CsvLoader(csv_data=csv_train, csv_val=csv_val,
                                 val_frac=val_frac, batch_size=batch_size)
    loaders = data_loaders.get_data_loaders()
    train_loader, val_loader = loaders

    info = f'CUSTOM_{model_name}'
    model = load_densenet(info, batch_size)
    df_mp, df_progress = model.fit(train_loader,
                                   val_loader,
                                   epochs=epochs)

def predict_densenet(model_name, csv_pred):
    """
    A function to allow for quickly generating prediction using densenet
    after a model has been trained.

    Parameters
    ----------
    model_name : str
        The name of your trained model.
    csv_pred : str
        The name of a .csv file located in "data/user_properties". This is
        the data you are predicting. This file needs to have two columns:
        "formula" and "target".

    Returns
    -------
    input_values : list
        List containing original 'target' values.
    predicted_values : list
        List containing predicted values.
    """
    mat_props_root = r'data/user_properties/'
    csv_pred = f'{mat_props_root}{csv_pred}'
    batch_size = 2**10
    data_loaders = EDM_CsvLoader(csv_data=csv_pred, batch_size=batch_size)
    loaders = data_loaders.get_data_loaders(inference=True)
    data_loader = loaders

    pretrained_models = cons.mps
    if model_name in pretrained_models:
        info = f'{model_name}'
        weights_pth = ('data/user_properties/trained_weights/'
                       f'DenseNet-{model_name}.pth')
    else:
        info = f'CUSTOM_{model_name}'
        weights_pth = ('data/user_properties/trained_weights/'
                       f'DenseNet-CUSTOM_{model_name}.pth')

    model = load_densenet(info, batch_size)
    compute_device = get_compute_device()
    checkpoint = torch.load(weights_pth, map_location=compute_device)

    model.load_checkpoint(checkpoint, model_name)
    input_values, predicted_values = model.predict(data_loader)
    return (input_values, predicted_values)


# %%
if __name__ == '__main__':
    # %%
    # =========================================================================
    #    Train densenet:
    #        Train on the desired property by:
    #            1. Naming your model (model_name='use_letters_or_underscores')
    #            2. Supplying a csv file with two columns 'formula', 'target'.
    #        The csv should be placed in the folder "data/user_properties".
    #        Your trained model weights will be saved in
    #       'data/user_properties/trained_weights'
    # =========================================================================
    train_densenet(model_name='your_model_name',
                   csv_train='example_bulk_train.csv')


# %%
    # =========================================================================
    #    Predict with densenet:
    #        Once a model has been trained, you simply need to give the model
    #        name and the csv file you want to predict on.
    #
    #        NOTE: the csv file requires the same formate as "train.csv".
    #        The 'target' values can be filled with any values (You can fill
    #        the target column with zeros if you are predicting compounds with
    #        unkown target values)
    # =========================================================================
    # print out all saved models to console
    list_saved_models()

    # get prediction for model of choice.
    prediction = predict_densenet(model_name='your_model_name',
                                  csv_pred='example_bulk_pred.csv')

    # predicted values are returned here.
    input_values, predicted_values = prediction
