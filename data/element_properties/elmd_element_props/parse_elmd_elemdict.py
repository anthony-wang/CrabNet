import json
import os
import numpy as np
import pandas as pd


# %%
with open("ElMD_ElementDict_elem_prop.json", 'r') as j:
    ed = json.loads(j.read())


# %%
out_dir = 'elmd_element_props'
ignore_list = ['mendeleev',
               'petti',
               'atomic',
               'mod_petti',
               'random_200',
               'magpie',
               'mat2vec',
               'jarvis',
               'oliynyk']


# %%
for elem_prop in ed.keys():
    if elem_prop in ignore_list or '_sc' in elem_prop:
        continue
    df = pd.DataFrame.from_dict(ed[elem_prop], orient='index')
    fname = f'{out_dir}/{elem_prop}.csv'
    df.to_csv(fname, index_label='element')
