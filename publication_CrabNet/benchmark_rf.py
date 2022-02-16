import os
import numpy as np
import warnings
import json

from time import time
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

from utils.get_core_count import get_core_count
from utils.modelselectionhelper import modelselectionhelper
from utils.utils import NumpyEncoder, CONSTANTS, count_gs_param_combinations


# %%
cons = CONSTANTS()
mat_props_dir = r'data/benchmark_data/'
mat_props = cons.benchmark_props

mat_props_names = cons.benchmark_names
mat_props_pretty = cons.benchmark_names_dict
elem_props = cons.eps
elem_props = ['magpie']

RNG_SEED = 42
np.random.seed(RNG_SEED)


# %%
models1 = {
    'RandomForestRegressor': RandomForestRegressor(),
}

rf_estimators_range = [500]

params1 = {
    'RandomForestRegressor': {'n_estimators': rf_estimators_range,
                              'max_depth': [None]},
}

scorings = {'r2': 'r2',
            'neg_MAE': 'neg_mean_absolute_error',
            'neg_RMSE': 'neg_root_mean_squared_error'
}


# %%
if __name__ == '__main__':
    start_datetime_benchmark_classics = (datetime.now()
                                     .strftime('%Y-%m-%d-%H%M%S.%f'))

    metrics_dir = f'metrics/rf_gridsearch/'
    fig_dir = r'figures/GridSearchCV/benchmark/'
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    _, cnt_dict = count_gs_param_combinations(params1)
    print(f'Number of parameter combinations for each estimator:\n'
          f'{cnt_dict}')

    cnt_dict_filename = 'parameter_combos_benchmark.json'
    cnt_dict_file = os.path.join(metrics_dir,
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
    n_jobs = n_cores // 2 - 2

    ignore_warnings = True
    if ignore_warnings:
        maxiter_msg = ('Maximum number of iteration reached '
                       'before convergence. Consider increasing max_iter '
                       'to improve the fit.')
        warnings.filterwarnings('ignore', message=maxiter_msg)

    ti_benchmark_classics = time()

    cv_folds = 2
    mshelper1 = modelselectionhelper(models1,
                                     params1,
                                     elem_props,
                                     mat_props_dir,
                                     mat_props,
                                     metrics_dir,
                                     fig_dir,
                                     scoring=scorings,
                                     n_jobs=n_jobs,
                                     cv=cv_folds,
                                     refit='neg_MAE',
                                     verbose=True,
                                     random_seed=RNG_SEED)
    dt_benchmark_classics = time() - ti_benchmark_classics

    print('*********** benchmark_classics finished ***********')
    print(f'benchmark_classics finished, elapsed time: '
          f'{dt_benchmark_classics:0.4g} s')
    print('*********** benchmark_classics finished ***********')
