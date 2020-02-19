import os

import numpy as np

import warnings

import json

from time import time
from datetime import datetime

from collections import OrderedDict

from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from utils.get_core_count import get_core_count
from utils.modelselectionhelper import modelselectionhelper
from utils.utils import NumpyEncoder, CONSTANTS


# %%
cons = CONSTANTS()
mat_props_dir = r'data/material_properties/'
mat_props = cons.mps

mat_props_names = cons.mp_names
mat_props_pretty = cons.mp_names_dict
elem_props = cons.eps


# %%
model_names_pretty = cons.classic_models_dict

models1 = {
    'Ridge': Ridge(),
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR()
}


ridge_alpha_range = list(np.logspace(-3, 3, num=7, base=10))

et_estimators_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
rf_estimators_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
ab_estimators_range = [16, 32, 64, 128, 256, 512]
ab_lr_range = [0.001, 0.01, 0.1, 1, 10, 100]

gb_estimators_range = [16, 32, 64, 128, 256, 512, 1024, 2048]
gb_lr_range = [0.001, 0.01, 0.1, 1]

svr_c_range = [0.1, 1, 10, 100, 1000, 10000, 100000]
svr_gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]


params1 = {
    'Ridge': {'alpha': ridge_alpha_range,
              'fit_intercept': [True]},
    'ExtraTreesRegressor': {'n_estimators': et_estimators_range,
                            'max_features': ['sqrt']},
    'RandomForestRegressor': {'n_estimators': rf_estimators_range,
                            'max_features': ['sqrt']},
    'AdaBoostRegressor': {'n_estimators': ab_estimators_range,
                          'learning_rate': ab_lr_range},
    'GradientBoostingRegressor': {'n_estimators': gb_estimators_range,
                                  'learning_rate': gb_lr_range},
    'KNeighborsRegressor': {'n_neighbors': [1],
                            'weights': ['uniform']},
    'SVR': {'C': svr_c_range,
            'gamma': svr_gamma_range}
}


scorings = {'r2': 'r2',
            'neg_MAE': 'neg_mean_absolute_error',
            'neg_RMSE': 'neg_root_mean_squared_error'
}

fig_dir = r'figures/GridSearchCV/'


# %%
if __name__ == '__main__':
    start_datetime_train_classics = (datetime.now()
                                     .strftime('%Y-%m-%d-%H%M%S.%f'))

    out_dir = (f'data/score_summaries/Classics/'
               f'Run_publication/')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    def count_param_combinations(d):
        cnt_dict = OrderedDict({})
        # array = []
        if (isinstance(d, (list))
            and not isinstance(d, (bool))):
            return len(d), cnt_dict
        elif (isinstance(d, (int, float, complex))
              and not isinstance(d, (bool))):
            return 1, cnt_dict
        elif (isinstance(d, (bool))
              or isinstance(d, (str))):
            return 1, cnt_dict
        elif isinstance(d, (dict, OrderedDict)):
            keys = d.keys()
            for k in keys:
                array = []
                subd = d[k]
                array.append(count_param_combinations(subd)[0])
                cnt = np.prod(array)
                cnt_dict[k] = cnt
            return np.prod(list(cnt_dict.values())), cnt_dict
        return cnt, cnt_dict

    _, cnt_dict = count_param_combinations(params1)
    print(f'Number of parameter combinations for each estimator:\n'
          f'{cnt_dict}')

    cnt_dict_filename = 'parameter_combos.json'
    cnt_dict_file = os.path.join(out_dir,
                                 cnt_dict_filename)
    json_content = json.dumps(cnt_dict,
                              cls=NumpyEncoder,
                              indent=4)

    with open(cnt_dict_file, 'w') as f:
        try:
            f.write(json_content)
        except:
            pass

    n_cores = get_core_count()
    n_jobs = n_cores // 2 - 1

    ignore_warnings = True

    if ignore_warnings:
        maxiter_msg = ('Maximum number of iteration reached '
                       'before convergence. Consider increasing max_iter '
                       'to improve the fit.')
        warnings.filterwarnings('ignore', message=maxiter_msg)

    ti_train_classics = time()

    cv_folds = 3
    mshelper1 = modelselectionhelper(models1,
                                     params1,
                                     elem_props,
                                     mat_props_dir,
                                     mat_props,
                                     out_dir,
                                     scoring=scorings,
                                     n_jobs=n_jobs,
                                     cv=cv_folds,
                                     refit='neg_MAE',
                                     verbose=True,
                                     random_seed=42)
    dt_train_classics = time() - ti_train_classics

    print('*********** train_classics finished ***********')
    print(f'train_classics finished, elapsed time: {dt_train_classics:0.4f} s')
    print('*********** train_classics finished ***********')
