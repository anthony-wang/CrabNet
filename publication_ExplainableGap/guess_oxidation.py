import os
import sys
import json
import traceback

import pandas as pd

if sys.platform == 'linux':
    # use parallel-processing Pandas
    # NOTE: only works on Linux/macOS or Windows under WSL
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False, nb_workers=8, use_memory_fs=True)

from pymatgen.core import Element

from tqdm import tqdm

from utils.utils import CONSTANTS
from utils.get_compute_device import get_compute_device

from utils.oxidation_utils import guess_oxidation, get_contents

compute_device = get_compute_device(prefer_last=True)
proc_device = get_compute_device(force_cpu=False)

cons = CONSTANTS()
elements = cons.atomic_symbols[1:]


# %%
bm_data_dir = 'data/benchmark_data'
bm_mat_props = os.listdir(bm_data_dir)

mb_data_dir = 'data/matbench_cv'
mb_mat_props = os.listdir(mb_data_dir)


# %% export oxidation states of the atoms
columns = ['element', 'common_oxidation_states',
           'icsd_oxidation_states', 'oxidation_states']
df_elem = pd.DataFrame(columns=columns)

for atom in elements:
    elem = Element(atom)
    common = elem.common_oxidation_states
    icsd = elem.icsd_oxidation_states
    all_states = elem.oxidation_states

    row_dict = {'element': atom,
                'common_oxidation_states': common,
                'icsd_oxidation_states': icsd,
                'oxidation_states': all_states,}
    df_elem = df_elem.append(row_dict, ignore_index=True)

outfile = 'data/oxidation/element_ox_states.json'
df_elem.index = df_elem['element']
df_elem.to_json(outfile, orient='columns')


# %%
os.makedirs('data/oxidation', exist_ok=True)
os.makedirs('data/oxidation/formulae', exist_ok=True)
os.makedirs('data/oxidation/guesses', exist_ok=True)


# %% guess oxidation using pymatgen package (note, this takes over several days)
for mat_prop in tqdm(bm_mat_props, desc='Processing benchmark data'):
    path = rf'data/benchmark_data/{mat_prop}/test.csv'
    df = pd.read_csv(path)
    formulae = df['formula']
    formulae.to_csv(f'data/oxidation/formulae/formulae_{mat_prop}.csv', index=False)
    try:
        if sys.platform == 'linux':
            ox = df['formula'].parallel_map(guess_oxidation)
        else:
            ox = df['formula'].map(guess_oxidation)
        ox = ox.rename('oxidation_states')
        ox.to_csv(f'data/oxidation/guesses/oxidation_{mat_prop}.csv', index=False)
        df_out = pd.concat([df, ox], axis=1)
        df_out.to_csv(f'data/oxidation/formulae+ox_{mat_prop}.csv', index=False)
        ox_states = df_out['oxidation_states']
        ox_states = ox_states.where(ox_states != '[]', other=None)
        ox_states = {form: get_contents(x, form) for x, form in zip(ox, df['formula'])}
        with open(f'data/oxidation/formulae+ox_{mat_prop}.json', 'w') as f:
            json.dump(ox_states, f, indent=2)
    except (Exception, ValueError):
        traceback.print_exc()
        pass


# %% guess oxidation using pymatgen package (note, this takes over several days)
for mat_prop in tqdm(mb_mat_props, desc='Processing matbench data'):
    for cv in tqdm([0,1,2,3,4], desc='Processing cv'):
        path = rf'data/matbench_cv/{mat_prop}/test{cv}.csv'
        df = pd.read_csv(path)
        formulae = df['formula']
        formulae.to_csv(f'data/oxidation/formulae/formulae_{mat_prop}{cv}.csv', index=False)
        try:
            if sys.platform == 'linux':
                ox = df['formula'].parallel_map(guess_oxidation)
            else:
                ox = df['formula'].map(guess_oxidation)
            ox = ox.rename('oxidation_states')
            ox.to_csv(f'data/oxidation/guesses/oxidation_{mat_prop}{cv}.csv', index=False)
            df_out = pd.concat([df, ox], axis=1)
            df_out.to_csv(f'data/oxidation/formulae+ox_{mat_prop}{cv}.csv', index=False)
            ox_states = df_out['oxidation_states']
            ox_states = ox_states.where(ox_states != '[]', other=None)
            ox_states = {form: get_contents(x, form) for x, form in zip(ox, df['formula'])}
            with open(f'data/oxidation/formulae+ox_{mat_prop}{cv}.json', 'w') as f:
                json.dump(ox_states, f, indent=2)
        except (Exception, ValueError):
            traceback.print_exc()
            pass

