import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore',
                        message='PyTorch is not compiled with NCCL support')

import torch

import matplotlib.pyplot as plt

import json

from time import time
from datetime import datetime

from collections import OrderedDict

from models.data import EDMDataLoader
from models.neuralnetwrapper import NeuralNetWrapper

from utils.utils import NumpyEncoder, CONSTANTS

plt.rcParams.update({'font.size': 16})


# %%
cons = CONSTANTS()
mat_props = cons.mps
mat_props_names = cons.mp_names
mat_props_pretty = cons.mp_names_dict

model_types = ['CrabNet', 'DenseNet']


# %%
if __name__ == '__main__':
    mat_props_root = r'data/material_properties/'

    orig_batch_size = 2**10
    epochs = 1001

    hidden_dims = [256, 128, 64]
    output_dims = 1

    # Seed random number generators
    manual_seeds = {7, 9, 2, 15, 4, 0, 10, 1337, 11, 13}
    RNG_SEEDS = sorted(list(manual_seeds))
    RNG_SEEDS = [9]

    ti_train_ann = time()
    start_datetime_train_ann = datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')

    out_dir = f'data/score_summaries/NN/Run_publication/'
    os.makedirs(out_dir, exist_ok=True)

    elem_prop = 'element'
    df_all = pd.DataFrame()

    for seed in RNG_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        for model_type in model_types:
            print(f'model_type: {model_type}')
            print(f'seed: {seed}')

            for mat_prop in mat_props:
                df_mp = pd.DataFrame()
                mat_prop_dir = mat_props_root + mat_prop

                if (mat_prop == 'energy_atom'
                    or mat_prop == 'Egap'
                    or mat_prop == 'oqmd'):
                    batch_size = max(orig_batch_size * 4, 2**12)
                else:
                    batch_size = orig_batch_size

                edm = (elem_prop == 'element'
                       or elem_prop == 'element_remix')

                print(f'featurizing {mat_prop} '
                      f'data using {elem_prop}')

                if elem_prop == 'element':
                    data_loaders = EDMDataLoader(mat_prop_dir,
                                                 elem_prop=elem_prop,
                                                 batch_size=batch_size)

                loaders = data_loaders.get_data_loaders()
                train_loader, val_loader, test_loader = loaders
                example_data = train_loader.dataset.data[0]
                input_dims = example_data.shape[-1]

                print(f'dataset: {train_loader.dataset}')
                print(f'dataset shape: {example_data.shape}')

                model = NeuralNetWrapper(model_type,
                                         elem_prop,
                                         mat_prop,
                                         input_dims,
                                         hidden_dims,
                                         output_dims,
                                         out_dir,
                                         edm=edm,
                                         batch_size=batch_size,
                                         random_seed=seed)

                df_mp, df_progress = model.fit(train_loader,
                                               val_loader,
                                               epochs=epochs)

                df_mp['rng_seed'] = seed

                addn_info_dict = OrderedDict({
                    'rng_seed': int(seed),
                    'torch_initial_seed': torch.random.initial_seed(),
                    'torch_get_rng_state': (torch.random.get_rng_state()
                                            .cpu().detach().numpy()),
                    'np_get_state': np.random.get_state(),
                    })

                addn_info_filename = str(df_mp['id'].iloc[-1]) + '.json'
                addn_info_file = os.path.join(out_dir,
                                              model.sub_dir,
                                              addn_info_filename)
                json_content = json.dumps(addn_info_dict,
                                          cls=NumpyEncoder,
                                          indent=4)

                with open(addn_info_file, 'w') as f:
                    try:
                        f.write(json_content)
                    except:
                        pass

                df_mp_filename = 'performance.csv'
                df_mp_file = os.path.join(out_dir,
                                          model.sub_dir,
                                          df_mp_filename)
                df_mp.to_csv(df_mp_file, index=False)

                df_progress_filename = 'progress.csv'
                df_progress_file = os.path.join(out_dir,
                                                model.sub_dir,
                                                df_progress_filename)
                df_progress.to_csv(df_progress_file, index=False)

                df_all = df_all.append(df_mp)
                df_all_filename = (f'all_results_Run_'
                                   + f'{start_datetime_train_ann}.csv')
                df_all_file = os.path.join(out_dir, df_all_filename)
                df_all.to_csv(df_all_file, index=False)

    dt = time() - ti_train_ann
    print('*********** train_ann finished ***********')
    print(f'train_ann finished, elapsed time: {dt:0.4f} s')
    print('*********** train_ann finished ***********')
