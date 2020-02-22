import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore',
                        message='PyTorch is not compiled with NCCL support')

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from time import time
from datetime import datetime

from torch import nn

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models.data import Scaler, LogNormScaler, MeanLogNormScaler, DummyScaler
from models.ann import DenseNet
from models.crabnet import CrabNet

from utils.utils import plot_training_curves
from utils.utils import plot_pred_act
from utils.utils import count_parameters
from utils.utils import xstrh
from utils.get_compute_device import get_compute_device

plt.rcParams.update({'font.size': 16})


# %%
class NeuralNetWrapper():
    def __init__(self,
                 model_type,
                 elem_prop,
                 mat_prop,
                 input_dims,
                 hidden_dims,
                 output_dims,
                 out_dir,
                 edm=False,
                 batch_size=1,
                 random_seed=None,
                 save_network_info=True):
        super(NeuralNetWrapper, self).__init__()

        self.model_type = model_type
        self.elem_prop = elem_prop
        self.mat_prop = mat_prop

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.out_dir = out_dir
        self.edm = edm
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.data_type = torch.float

        self.save_network_info = save_network_info

        # Default to using the last GPU available
        self.CUDA_count = torch.cuda.device_count()
        self.compute_device = get_compute_device()

        print(f'Creating Model of type {self.model_type}')
        if self.model_type == 'CrabNet':
            self.model = CrabNet(self.compute_device,
                                 input_dims=self.input_dims,
                                 d_model=64,
                                 nhead=4,
                                 num_layers=2,
                                 dim_feedforward=16,
                                 dropout=0.3,
                                 edm=self.edm)
        elif self.model_type == 'DenseNet':
            self.model = DenseNet(self.compute_device,
                                 input_dims=self.input_dims,
                                 hidden_dims=self.hidden_dims,
                                 output_dims=self.output_dims,
                                 dropout=0.1,
                                 edm=self.edm)

        self.model.to(self.compute_device,
                      dtype=self.data_type,
                      non_blocking=True)

        self.num_network_params = count_parameters(self.model)
        print(f'number of network params: {self.num_network_params}')

        self.criterion = nn.MSELoss()
        self.optim_lr = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.optim_lr)

        # Logging
        self.start_time = datetime.now()
        self.start_datetime = self.start_time.strftime('%Y-%m-%d-%H%M%S.%f')
        self.log_filename = (f'{self.start_datetime}-{self.model_type}-'
                             f'{self.elem_prop}-{self.mat_prop}.log')
        self.sub_dir = (f'{self.start_datetime}-{xstrh(self.random_seed)}'
                        f'{self.model_type}-'
                        f'{self.elem_prop}-{self.mat_prop}')
        self.log_dir = os.path.join(self.out_dir, self.sub_dir)

        if 'CUSTOM' in self.mat_prop:
            os.makedirs(self.log_dir, exist_ok=True)

        if self.save_network_info:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.out_dir,
                                         self.sub_dir ,
                                         self.log_filename)

            print(56 * '*')
            print(f'creating and writing to log file {self.log_file}')
            print(56 * '*')
            with open(self.log_file, 'a') as f:
                try:
                    f.write('Start time: ')
                    f.write(f'{self.start_datetime}\n')
                    f.write(f'random seed: {self.random_seed}\n')
                    f.write('Model type: ')
                    f.write(f'{self.model_type}\n')
                    f.write('Material property: ')
                    f.write(f'{self.mat_prop}\n')
                    f.write('Element property: ')
                    f.write(f'{self.elem_prop}\n')
                    f.write(f'EDM input: {self.edm}\n')
                    f.write('Network architecture:\n')
                    f.write(f'{self.model}\n')
                    f.write(f'Number of params: ')
                    f.write(f'{self.num_network_params}\n')
                    f.write(f'CUDA count: {self.CUDA_count}\n')
                    f.write(f'Compute device: {self.compute_device}\n')
                    f.write('Criterion and Optimizer:\n')
                    f.write(f'{self.criterion}\n')
                    f.write(f'{self.optimizer}\n')
                    f.write(56 * '*' + '\n')
                except:
                    pass


    def get_target_scaler(self, target, mat_prop):
        if (mat_prop == 'agl_thermal_conductivity_300K'
            or mat_prop == 'ael_debye_temperature'):
            target_scaler = MeanLogNormScaler(target)
        else:
            target_scaler = Scaler(target)

        return target_scaler


    def fit(self, train_loader, val_loader, epochs=1001):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs

        ti_fit = time()

        epoch_times = []
        cumulative_times = []

        mse_trains = []
        mae_trains = []
        r2_trains = []
        mse_vals = []
        mae_vals = []
        r2_vals = []

        mean_t_r2 = np.nan
        mean_t_mae = np.nan
        mean_v_r2 = np.nan
        mean_v_mae = np.nan
        std_t_r2 = np.nan
        std_t_mae = np.nan
        std_v_r2 = np.nan
        std_v_mae = np.nan

        r2_train_max = float('-inf')
        mae_train_max = float('inf')
        r2_val_max = float('-inf')
        mae_val_max = float('inf')

        y_train = [data[1].numpy().tolist() for data in self.train_loader]
        y_train = [item for sublist in y_train for item in sublist]

        self.target_scaler = self.get_target_scaler(y_train, self.mat_prop)

        print(f'Fitting neural network {self.model_type}...')
        if self.save_network_info:
            with open(self.log_file, 'a') as f:
                try:
                    f.write(f'Datasets (batch size {self.batch_size}):\n')
                    f.write(f'Train {train_loader.dataset}\n')
                    f.write(f'Val {val_loader.dataset}\n')
                except:
                    pass

        for epoch in range(self.epochs):
            ti_epoch = time()

            for i, data_output in enumerate(self.train_loader):
                ti_batch = time()
                X, y, formulae = data_output

                # Scale target values
                y = self.target_scaler.scale(y)

                X = X.to(self.compute_device,
                         dtype=self.data_type,
                         non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=self.data_type,
                         non_blocking=True)

                self.optimizer.zero_grad()
                output = self.model.forward(X).flatten()
                loss = self.criterion(output.view(-1), y.view(-1))
                loss.backward()
                self.optimizer.step()

                dt_batch = time() - ti_batch
                training_rate = round(X.shape[0] / dt_batch)

            if epoch % 1 == 0:
                # Get train targets and predictions
                ti_predict = time()
                target_train, pred_train = self.predict(self.train_loader)
                dt_predict = time() - ti_predict
                prediction_rate = round(len(self.train_loader.dataset)
                                        / dt_predict)
                print(f'prediction rate: '
                      f'{prediction_rate} '
                      f'samples/second')

                # Get val targets and predictions
                target_val, pred_val = self.predict(self.val_loader)

                # Append stats
                r2_trains.append(r2_score(target_train, pred_train))
                mse_trains.append(mean_squared_error(target_train,
                                                     pred_train))
                mae_trains.append(mean_absolute_error(target_train,
                                                      pred_train))
                r2_vals.append(r2_score(target_val, pred_val))
                mse_vals.append(mean_squared_error(target_val, pred_val))
                mae_vals.append(mean_absolute_error(target_val, pred_val))

                # Get best results so far
                if r2_trains[-1] > r2_train_max:
                    r2_train_max = r2_trains[-1]
                if mae_trains[-1] < mae_train_max:
                    mae_train_max = mae_trains[-1]
                if r2_vals[-1] > r2_val_max:
                    r2_val_max = r2_vals[-1]
                if mae_vals[-1] < mae_val_max:
                    mae_val_max = mae_vals[-1]

                # Calculate running mean and std
                if epoch > 19:
                    mean_t_r2 = np.mean(r2_trains[-20:])
                    mean_t_mae = np.mean(mae_trains[-20:])
                    mean_v_r2 = np.mean(r2_vals[-20:])
                    mean_v_mae = np.mean(mae_vals[-20:])
                    std_t_r2 = np.std(r2_trains[-20:])
                    std_t_mae = np.std(mae_trains[-20:])
                    std_v_r2 = np.std(r2_vals[-20:])
                    std_v_mae = np.std(mae_vals[-20:])

                # Calculate difference in train and val metrics
                diff_r2 = r2_vals[-1] - r2_trains[-1]
                diff_mae = mae_trains[-1] - mae_vals[-1]

                print(56 * '-')
                print(f'net: {self.model_type}, '
                      f'mat_prop: {self.mat_prop}, '
                      f'epoch: {epoch:d}, '
                      f'lr: {self.optim_lr:0.2e}')

                print(f'r2 train score: {r2_trains[-1]:0.4f} '
                      f'(best: {r2_train_max:0.4f}, '
                      f'last 20 avg: {mean_t_r2:0.4f}, '
                      f'std: {std_t_r2:0.4f})')
                print(f'r2 val score: {r2_vals[-1]:0.4f} '
                      f'(best: {r2_val_max:0.4f}, '
                      f'last 20 avg: {mean_v_r2:0.4f}, '
                      f'std: {std_v_r2:0.4f})')
                print(f'difference in r2: {diff_r2:0.4f}')

                print(f'mae train score: {mae_trains[-1]:0.4f} '
                      f'(best: {mae_train_max:0.4f}, '
                      f'last 20 avg: {mean_t_mae:0.4f}, '
                      f'std: {std_t_mae:0.4f})')
                print(f'mae val score: {mae_vals[-1]:0.4f} '
                      f'(best: {mae_val_max:0.4f}, '
                      f'last 20 avg: {mean_v_mae:0.4f}, '
                      f'std: {std_v_mae:0.4f})')
                print(f'difference in mae: {diff_mae:0.4f}')

                print('- - - -')
                print(f'batch time: {dt_batch:0.4f} s, '
                      f'batch size: {self.batch_size}')
                print(f'training rate: '
                      f'{training_rate} '
                      f'samples/second')

            dt_epoch = time() - ti_epoch
            print(f'1 epoch time: {dt_epoch:0.4f} s '
                  f'with {self.num_network_params} params on '
                  f'{self.compute_device}')
            print(f'time left: {(epochs - epoch) * dt_epoch:0.2f} s')
            epoch_times.append(dt_epoch)

            if len(cumulative_times) == 0:
                cumulative_times.append(dt_epoch)
            else:
                cumulative_times.append(cumulative_times[-1] + dt_epoch)
            if self.save_network_info:
                with open(self.log_file, 'a') as f:
                    try:
                        f.write(56 * '*' + '\n')
                        f.write(f'net: {self.model_type}, '
                                f'epoch: {epoch:d}, '
                                f'lr: {self.optim_lr:0.2e}\n')

                        f.write(f'r2 train score: {r2_trains[-1]:0.4f} '
                                f'(best: {r2_train_max:0.4f}, '
                                f'last 20 avg: {mean_t_r2:0.4f}, '
                                f'std: {std_t_r2:0.4f})\n')
                        f.write(f'r2 val score: {r2_vals[-1]:0.4f} '
                                f'(best: {r2_val_max:0.4f}, '
                                f'last 20 avg: {mean_v_r2:0.4f}, '
                                f'std: {std_v_r2:0.4f})\n')
                        f.write(f'difference in r2: {diff_r2:0.4f}\n')

                        f.write(f'mae train score: {mae_trains[-1]:0.4f} '
                                f'(best: {mae_train_max:0.4f}, '
                                f'last 20 avg: {mean_t_mae:0.4f}, '
                                f'std: {std_t_mae:0.4f})\n')
                        f.write(f'mae val score: {mae_vals[-1]:0.4f} '
                                f'(best: {mae_val_max:0.4f}, '
                                f'last 20 avg: {mean_v_mae:0.4f}, '
                                f'std: {std_v_mae:0.4f})\n')
                        f.write(f'difference in mae: {diff_mae:0.4f}\n')

                        f.write(f'batch time: {dt_batch:0.4f} s, '
                                f'batch size: {self.batch_size}\n')
                        f.write(f'training rate: '
                                f'{training_rate} '
                                f'samples/second\n')
                        f.write(f'prediction rate: '
                                f'{prediction_rate} '
                                f'samples/second\n')
                    except:
                        pass

            if r2_val_max > 0.4 and mae_val_max == mae_vals[-1]:
                print('Saving model checkpoint')
                self.pth_filename = ('checkpoint.pth')
                self.pth_file = os.path.join(self.out_dir,
                                             self.sub_dir ,
                                             self.pth_filename)

                save_dict = {'weights': self.model.state_dict(),
                             'scaler_state': self.target_scaler.state_dict()}

                path = self.pth_file
                if 'CUSTOM' in self.mat_prop:
                    print('custom model found')

                    path = ('data/user_properties/trained_weights/'
                            f'{self.model_type}-'
                            f'{self.mat_prop}.pth')

                torch.save(save_dict, path)

                if self.save_network_info:
                    with open(self.log_file, 'a') as f:
                        try:
                            f.write(56 * '#' + '\n')
                            f.write(f'New r2_val record reached at epoch '
                                    f'{epoch},\n'
                                    f'model checkpoint saved as '
                                    f'{self.pth_filename}\n')
                        except:
                            pass

            if epoch % 50 == 0 or epoch in [3, 5, 10, 20, 30, 40, epochs-1]:
                # Plot training curve
                fig = plot_training_curves(
                            mae_trains,
                            mse_trains,
                            r2_trains,
                            mae_vals,
                            mse_vals,
                            r2_vals,
                            mae_val_max,
                            r2_val_max,
                            self.model_type,
                            epoch,
                            self.elem_prop,
                            self.mat_prop,
                            self.train_loader.dataset,
                            type(self.optimizer))
                plt.close('all')
                fig_file = os.path.join(self.log_dir,
                                        f'epoch{epoch}-train_curve.png')
                fig.savefig(fig_file,
                            dpi=300,
                            bbox_inches='tight')

                # Do full eval pass, report stats
                # Get test targets and predictions
                _, _, r2_val = self.evaluate(target_val, pred_val)

                # Plot predicted vs. actual curve
                fig = plot_pred_act(target_val,
                                    pred_val,
                                    epoch,
                                    addn_title_text=f'r2_val: {r2_val:0.4f}',
                                    label=self.mat_prop,
                                    outliers=False,
                                    threshold=0.5)
                plt.close('all')
                fig_file = os.path.join(self.log_dir,
                                        f'epoch{epoch}-pred_act.png')
                fig.savefig(fig_file,
                            dpi=300,
                            bbox_inches='tight')

        self.dt_fit = time() - ti_fit
        self.end_time = datetime.now()
        self.end_datetime = self.end_time.strftime('%Y-%m-%d-%H%M%S.%f')
        print(f'total fitting time: {self.dt_fit:0.4f} s')

        # load state_dict and evaluate
        if 'CUSTOM' not in self.mat_prop:
            if hasattr(self, 'pth_file'):
                print(56 * '-')
                print(56 * '-')
                print(f'loading best trained network and evaluating valset')

                checkpoint = torch.load(self.pth_file)
                test_output = self.evaluate_checkpoint(checkpoint,
                                                       self.val_loader)
                mae_val, mse_val, r2_val = test_output
                print(f'r2 val score: {r2_val:0.4f}\n'
                      f'mae val score: {mae_val:0.4f}\n'
                      f'mse val score: {mse_val:0.4f}\n')

                target_val, pred_val = self.predict(self.val_loader)
                _, _, r2_val = self.evaluate(target_val, pred_val)
                fig = plot_pred_act(target_val,
                                    pred_val,
                                    epoch=None,
                                    addn_title_text=f'best r2_val: {r2_val:0.4f}',
                                    label=self.mat_prop,
                                    outliers=False,
                                    threshold=0.5)
                plt.close('all')
                fig_file = os.path.join(self.log_dir,
                                        f'best-pred_act.png')
                fig.savefig(fig_file,
                            dpi=300,
                            bbox_inches='tight')

        if self.save_network_info:
            with open(self.log_file, 'a') as f:
                try:
                    f.write(56 * '*' + '\n')
                    f.write(f'fitting finished at {self.end_datetime}\n')
                    f.write(f'total fitting time: {self.dt_fit:0.4f} s\n')
                    f.write(f'testing performance using best '
                            f'trained network on val:\n')
                    f.write(f'r2 val score: {r2_val:0.4f}\n'
                            f'mae val score: {mae_val:0.4f}\n'
                            f'mse val score: {mse_val:0.4f}\n')
                except:
                    pass

        df_progress_dict = {
            'epoch': np.arange(epochs),
            'epoch_times': epoch_times,
            'cumulative_times': cumulative_times,
            'mae_train': mae_trains,
            'mse_train': mse_trains,
            'r2_train': r2_trains,
            'mae_val': mae_vals,
            'mse_val': mse_vals,
            'r2_val': r2_vals
            }
        df_progress = pd.DataFrame.from_dict(df_progress_dict,
                                             orient='columns')

        df_fit_columns = ['id', 'model_type', 'num_network_params',
                          'elem_prop', 'mat_prop', 'epochs', 'fit_time',
                          'mae_train', 'mse_train', 'r2_train',
                          'mae_val', 'mse_val', 'r2_val']
        df_fit = pd.DataFrame(columns=df_fit_columns)

        best_mae_idx = mae_vals.index(min(mae_vals))
        df_fit_row = {
            'id': (f'{self.start_datetime}-{xstrh(self.random_seed)}'
                   f'{self.model_type}-'
                   f'{self.elem_prop}-{self.mat_prop}'),
            'model_type': self.model_type,
            'num_network_params': self.num_network_params,
            'elem_prop': self.elem_prop,
            'mat_prop': self.mat_prop,
            'epochs': self.epochs,
            'fit_time': self.dt_fit,
            'mae_train': mae_trains[best_mae_idx],
            'mse_train': mse_trains[best_mae_idx],
            'r2_train': r2_trains[best_mae_idx],
            'mae_val': mae_vals[best_mae_idx],
            'mse_val': mse_vals[best_mae_idx],
            'r2_val': r2_vals[best_mae_idx],
            }
        df_fit = df_fit.append(df_fit_row, ignore_index=True)

        return df_fit, df_progress


    def predict(self, data_loader):
        target_list = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for i, data_output in enumerate(data_loader):
                X_val, y_act_val, v_form = data_output
                X_val = X_val.to(self.compute_device,
                                 dtype=self.data_type,
                                 non_blocking=True)
                y_act_val = y_act_val.cpu().flatten().tolist()
                y_pred_val = self.model.forward(X_val).cpu().flatten().tolist()

                # Unscale target values
                y_pred_val = self.target_scaler.unscale(y_pred_val).tolist()

                targets = y_act_val
                predictions = y_pred_val
                target_list.extend(targets)
                pred_list.extend(predictions)
        self.model.train()

        return target_list, pred_list


    def evaluate(self, target, pred):
        r2 = r2_score(target, pred)
        mse = mean_squared_error(target, pred)
        mae = mean_absolute_error(target, pred)
        output = (mae, mse, r2)
        return output


    def load_checkpoint(self, checkpoint, mat_prop):
        self.target_scaler = self.get_target_scaler(torch.zeros(3), mat_prop)
        self.model.load_state_dict(checkpoint['weights'])
        self.target_scaler.load_state_dict(checkpoint['scaler_state'])


    def evaluate_checkpoint(self, checkpoint, data_loader):
        self.load_checkpoint(checkpoint, self.mat_prop)
        target, pred = self.predict(data_loader)
        output = self.evaluate(target, pred)
        return output
