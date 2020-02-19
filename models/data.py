import os

import pandas as pd
import numpy as np

import torch

from tqdm import tqdm

from utils.composition import generate_features, _element_composition

from sklearn.preprocessing import StandardScaler, Normalizer

from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict


# %%
def get_cbfv(path, elem_prop='oliynyk'):
    """
    Loads the compound csv file and featurizes it, then scales the features
    using StandardScaler.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(path)
    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]
    X, y, formula = generate_features(df, elem_prop)

    # scale each column of data to have a mean of 0 and a variance of 1
    scaler = StandardScaler()
    # normalize each row in the data
    normalizer = Normalizer()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(normalizer.fit_transform(X_scaled),
                            columns=X.columns.values,
                            index=X.index.values)

    return X_scaled, y, formula



# %%
def get_edm(path, elem_prop='onehot', n_elements=12):
    """
    Build a element descriptor matrix.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
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

    df = pd.read_csv(path)
    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]

    list_ohm = [OrderedDict(_element_composition(form))
                for form in df['formula']]
    list_ohm = [OrderedDict(sorted(mat.items(), key=lambda x:-x[1]))
                for mat in list_ohm]

    _, y, formula = generate_features(df, elem_prop)

    edm_array = np.zeros(shape=(len(list_ohm), n_elements, len(all_symbols)+1))
    for i, comp in enumerate(tqdm(list_ohm)):
        for j, (elem, count) in enumerate(list_ohm[i].items()):
            if j == n_elements:
                # Truncate EDM representation to n_elements
                break
            edm_array[i, j, all_symbols.index(elem) + 1] = count

    # Scale features
    for i in range(edm_array.shape[0]):
        edm_array[i, :, -1] = (edm_array[i, :, :].sum(axis=-1)
                               / (edm_array[i, :, :].sum(axis=-1)).sum())
        n_atoms = np.count_nonzero(edm_array[i, :, -1])

        # Scale the one-hot and fractional encodings
        one_hot_slice = edm_array[i, :, :-1][:n_atoms, :]
        slice_mean = np.mean(one_hot_slice)
        edm_array[i, :, :-1][:n_atoms, :] = one_hot_slice - slice_mean

    return edm_array, y, formula


# %%
class CBFVDataLoader():
    """
    Parameters
    ----------
    train_file: str
        name of csv file containing cif and properties
    val_file: str
        name of csv file containing cif and properties
    train_ratio: float, optional (default=0.75)
        train/val ratio if val_file not given
    batch_size: float, optional (default=64)
        Step size for the Gaussian filter
    random_state: int, optional (default=123)
        Random seed for sampling the dataset. Only used if validation data is
        not given.
    shuffle: bool (default=False)
        Whether to shuffle the datasets or not
    """
    def __init__(self, data_dir, elem_prop='oliynyk',
                 batch_size=64, num_workers=1, random_state=42,
                 shuffle=True, pin_memory=True):

        self.data_dir = data_dir
        self.elem_prop = elem_prop

        self.train_loc = os.path.join(data_dir, 'train.csv')
        self.val_loc = os.path.join(data_dir, 'val.csv')
        self.test_loc = os.path.join(data_dir, 'test.csv')

        self.train_data = get_cbfv(self.train_loc,
                                         elem_prop=self.elem_prop)
        self.val_data = get_cbfv(self.val_loc,
                                       elem_prop=self.elem_prop)
        self.test_data = get_cbfv(self.test_loc,
                                        elem_prop=self.elem_prop)

        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.shuffle = shuffle
        self.random_state = random_state

        self.n_comp = 8


    def get_data_loaders(self, batch_size=1, inference=False):
        '''
        Input the dataset, get train test split
        '''
        train_dataset = CBFVDataset(self.train_data, self.n_comp)
        val_dataset = CBFVDataset(self.val_data, self.n_comp)
        test_dataset = CBFVDataset(self.test_data, self.n_comp)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory,
                                shuffle=self.shuffle)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        return train_loader, val_loader, test_loader



# %%
class EDMDataLoader():
    """
    Parameters
    ----------
    train_file: str
        name of csv file containing cif and properties
    val_file: str
        name of csv file containing cif and properties
    train_ratio: float, optional (default=0.75)
        train/val ratio if val_file not given
    batch_size: float, optional (default=64)
        Step size for the Gaussian filter
    random_state: int, optional (default=123)
        Random seed for sampling the dataset. Only used if validation data is
        not given.
    shuffle: bool (default=False)
        Whether to shuffle the datasets or not
    """
    def __init__(self, data_dir, elem_prop='onehot',
                 batch_size=64, num_workers=1, random_state=42,
                 shuffle=True, pin_memory=True):

        self.data_dir = data_dir
        self.elem_prop = elem_prop

        self.train_loc = os.path.join(data_dir, 'train.csv')
        self.val_loc = os.path.join(data_dir, 'val.csv')
        self.test_loc = os.path.join(data_dir, 'test.csv')

        self.train_data = get_edm(self.train_loc, elem_prop='onehot')
        self.val_data = get_edm(self.val_loc, elem_prop='onehot')
        self.test_data = get_edm(self.test_loc, elem_prop='onehot')

        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.shuffle = shuffle
        self.random_state = random_state

        self.n_comp = 8


    def get_data_loaders(self, batch_size=1, inference=False):
        '''
        Input the dataset, get train test split
        '''
        train_dataset = EDMDataset(self.train_data, self.n_comp)
        val_dataset = EDMDataset(self.val_data, self.n_comp)
        test_dataset = EDMDataset(self.test_data, self.n_comp)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory,
                                shuffle=self.shuffle)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        return train_loader, val_loader, test_loader


# %%
class EDM_CsvLoader():
    """
    Parameters
    ----------
    csv_data: str
        name of csv file containing cif and properties
    csv_val: str
        name of csv file containing cif and properties
    val_frac: float, optional (default=0.75)
        train/val ratio if val_file not given
    batch_size: float, optional (default=64)
        Step size for the Gaussian filter
    random_state: int, optional (default=123)
        Random seed for sampling the dataset. Only used if validation data is
        not given.
    shuffle: bool (default=True)
        Whether to shuffle the datasets or not
    """

    def __init__(self, csv_data, csv_val=None, val_frac=0.2, batch_size=64,
                 num_workers=1, random_state=0, shuffle=False,
                 pin_memory=True):
        self.csv_data = csv_data
        self.csv_val = csv_val
        self.val_frac = val_frac
        self.main_data = list(get_edm(self.csv_data, elem_prop='onehot'))
        self.n_train = len(self.main_data[0])

        if self.csv_val is None:
            if self.val_frac > 0:
                self.n_val = int(self.n_train * self.val_frac)
            val_idx = np.random.choice(np.arange(self.n_train),
                                       self.n_val, replace=False)
            train_idx = np.isin(np.arange(self.n_train), val_idx, invert=True)
            self.val_data = [None, None, None]
            self.train_data = [None, None, None]
            for i in range(3):
                self.val_data[i] = self.main_data[i][val_idx]
                self.train_data[i] = self.main_data[i][train_idx]
        else:
            self.val_data = get_edm(self.csv_val, elem_prop='onehot')

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_comp = 8

    def get_data_loaders(self, inference=False):
        '''
        Input the dataset, get train test split
        '''

        if inference is True:
            pred_dataset = EDMDataset(self.main_data, self.n_comp)
            pred_loader = DataLoader(pred_dataset,
                                     batch_size=self.batch_size,
                                     pin_memory=self.pin_memory,
                                     shuffle=self.shuffle)
            return pred_loader

        train_dataset = EDMDataset(self.train_data, self.n_comp)
        val_dataset = EDMDataset(self.val_data, self.n_comp)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory,
                                shuffle=self.shuffle)

        return train_loader, val_loader



# %%
class CBFVDataset(Dataset):
    """
    Get X and y from CBFV-based dataset.
    """
    def __init__(self, dataset, n_comp):
        self.data = dataset
        self.n_comp = n_comp

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])

        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]

    def __str__(self):
        string = f'CBFVDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[[idx], :]
        y = self.y[idx]
        formula = self.formula[idx]

        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        return (X, y, formula)


# %%
class EDMDataset(Dataset):
    """
    Get X and y from EDM dataset.
    """
    def __init__(self, dataset, n_comp):
        self.data = dataset
        self.n_comp = n_comp

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])

        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]

    def __str__(self):
        string = f'EDMDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]

        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        return (X, y, formula)


# %%
class Scaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class LogNormScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = torch.log(data)
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = torch.exp(data_scaled)
        return data


class MeanLogNormScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.logdata = torch.log(self.data)
        self.mean = torch.mean(self.logdata)
        self.std = torch.std(self.logdata)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (torch.log(data) - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled) * self.std + self.mean
        data = torch.exp(data_scaled)
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class DummyScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        return torch.as_tensor(data)

    def unscale(self, data_scaled):
        return torch.as_tensor(data_scaled)

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# %%
if __name__ == '__main__':
    pass
