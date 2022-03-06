import os
import glob
import subprocess
import shutil

import zarr

from copy import copy

import traceback

import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time
from datetime import timedelta

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

import torch
from torch import nn

compute_device = get_compute_device(prefer_last=True)

# get environment variables
ON_CLUSTER = os.environ.get('ON_CLUSTER')
HOME = os.environ.get('HOME')
USER_EMAIL = os.environ.get('USER_EMAIL')


# %%
class SaveOutput:
    def __init__(self, N, steps, attn_shape, chunks_shape):
        # N is number of layers
        # steps is the total expected number of steps (epochs or minibatches)
        self.N = N
        self.outputs = [[] for _ in range(self.N)]
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = 0
        self.array_count = 0
        self.n = 0
        self.minibatches = len(model.data_loader)
        self.layer_names = [f'layer{layer}' for layer in range(self.N)]
        self.stores = [self.init_store(f'{data_save_path}/attn_data_layer{layer}.zarr')
                       for layer in range(self.N)]
        self.roots = [self.init_group(st) for st in self.stores]
        array_len = attn_shape[0] * steps
        self.array_shape = tuple([array_len] + list(attn_shape)[1:])
        self.attn_arrays = [self.init_array(
            self.roots[layer],
            self.layer_names[layer],
            self.array_shape,
            chunks_shape) for layer in range(self.N)]  # for each layer

        print(self.array_shape)
        print(self.attn_arrays)

    def __call__(self, module, module_in, module_out):
        # only capture output if requires_grad == False (i.e. in validation)
        # only capture output if requires_grad == True (i.e. in training)
        if model.capture_flag == True:
            # mod_out is a tuple
            # mod_out[0] is attn_output (softmax(Q.K^T))
            # mod_out[1] is attn_output_weights (this is what we need)
            self.outputs[self.n].append(module_out[1])
            self.formulae.append(model.formula_current)
            self.acts.append(model.act_v)
            self.preds.append(model.pred_v)

            self.counter += 1
            self.n += 1
            if self.n >= self.N:
                # if we have exceed the number of layers, then reset to 0
                # print('n exceeded, resetting to 0')
                self.n = 0
            if self.counter >= self.minibatches * self.N:
                # if we have accummulated enough tensors, then save
                self.save_zarr()
                print('saving to zarr array done, now clearing outputs')
                self.clear_outputs()
                self.counter = 0

    def init_store(self, path):
        store = zarr.NestedDirectoryStore(path)
        return store

    def init_group(self, store):
        group = zarr.group(store, overwrite=True)
        return group

    def init_array(self, group, name, array_shape, chunks_shape):
        array = group.full(name,
                           shape=array_shape,
                           chunks=chunks_shape,
                           fill_value='NaN',
                           dtype='float32',
                           compression=None,
                           overwrite=True
                          )
        return array

    def save_zarr(self):
        for L in range(self.N):
            mod_out = self.outputs[L]
            mat_dims = np.array([mat.shape[0] for mat in mod_out])
            n_train = len(model.train_loader.dataset)
            n_val = len(model.data_loader.dataset)
            n_total = n_train + n_val

            n_mats = len(mod_out)  # number of output matrices from hook
            bsz = model.data_loader.batch_size  # batch size from data loader

            B_train = len(model.train_loader)  # total number of batches from train data loader
            B_val = len(model.data_loader)  # total number of batches from val data loader
            B = B_train + B_val  # total number of batches from data loader

            # Collect batches from validation set
            B = B_val

            # number of minibatches in epoch
            num_mbs_in_epoch = len(model.data_loader)

            H = model.model.heads  # number of heads
            N = model.model.N  # number of layers

            n_data = n_val
            n_elements = model.n_elements

            if n_mats != B * model.epochs:
                epochs = n_mats // B

            attn_data = np.asarray(torch.cat(mod_out, dim=0).unsqueeze(1).cpu().detach())

            if self.attn_arrays[L].nchunks_initialized == 0:
                # if the array is empty, then fill the first chunk
                self.attn_arrays[L][:n_data] = attn_data
            elif self.attn_arrays[L].nchunks_initialized == B * model.epochs:
                # array is filled all the way
                continue
            else:
                # append new chunks continuously
                self.attn_arrays[L][self.array_count * n_data:
                                    (self.array_count + 1) * n_data] = attn_data
        self.array_count += 1

    def clear(self):
        self.outputs = [[] for _ in range(self.N)]
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = 0
        self.n = 0

    def clear_outputs(self):
        self.n_mats = len(self.outputs[0])
        self.outputs = [[] for _ in range(self.N)]
        self.counter = 0


# %%
if __name__ == '__main__':
    print('this is __main__!')
    print(f'{ON_CLUSTER = }')
    print(f'{HOME = }')
    print(f'{USER_EMAIL = }')

    t0_all = time()
    data_dir = 'data/benchmark_data'
    mat_props = os.listdir(data_dir)

    mat_props = ['aflow__Egap']

    for mat_prop in mat_props:
        print(f'currently on mat_prop: {mat_prop}')

        data_save_path_orig = f'data_save/{mat_prop}'
        data_save_path = copy(data_save_path_orig)

        if ON_CLUSTER:
            print('Running on CLUSTER!')
            prepend_path = '/scratch/awang/'
            data_save_path = prepend_path + data_save_path_orig
            print(f'{data_save_path = }')

        os.makedirs(data_save_path, exist_ok=True)

        t0_mp = time()

        # set the attention tensor capturing method, either every 'step' or 'epoch'
        capture_every = 'epoch'
        allowed_captures = ['step', 'epoch']

        err_msg = f'the "capture_every" keyword should be one of {allowed_captures}!'
        assert capture_every in allowed_captures, err_msg

        crabnet_params = {'d_model': 512, 'N': 3, 'heads': 4}
        model = Model(CrabNet(**crabnet_params, compute_device=compute_device).\
                      to(compute_device),
                      model_name=f'{mat_prop}',
                      capture_every=capture_every,
                      verbose=True)

        train_data = rf'{data_dir}/{mat_prop}/train.csv'
        val_data = rf'{data_dir}/{mat_prop}/val.csv'

        data_size = pd.read_csv(train_data).shape[0]
        batch_size = 2**round(np.log2(data_size)-4)
        if batch_size < 2**7:
            batch_size = 2**7
        if batch_size > 2**10:
            batch_size = 2**10
        model.load_data(train_data, batch_size=batch_size, train=True)
        model.load_data(val_data, batch_size=batch_size)
        print(f'training with batchsize {model.batch_size} '
              f'(2**{np.log2(model.batch_size):0.2f})')

        # Get pertinent information from model
        n_train = len(model.train_loader.dataset)
        n_val = len(model.data_loader.dataset)
        n_total = n_train + n_val

        bsz = model.data_loader.batch_size  # batch size from data loader

        B_train = len(model.train_loader)  # total number of batches from train data loader
        B_val = len(model.data_loader)  # total number of batches from val data loader
        B = B_train + B_val  # total number of batches from data loader

        # Collect batches from validation set
        B = B_val

        # number of minibatches in epoch
        num_mbs_in_epoch = len(model.data_loader)

        H = model.model.heads  # number of heads
        N = model.model.N  # number of layers

        n_data = n_val
        n_elements = model.n_elements
        # Get pertinent information from model

        # Set the number of epochs, decide if you want a loss curve to be plotted
        # 10-20 is a pretty good balance between time and first look IMO
        # CUDA out of memory if epochs = 50 if capturing every epoch!
        epochs = 50

        chunks_shape = [n_data, 1, H, n_elements, n_elements]
        chunks_shape = tuple(chunks_shape)
        attn_shape = [n_data, 1, H, n_elements, n_elements]
        attn_shape = tuple(attn_shape)

        # calculate number of steps
        if capture_every == 'epoch':
            steps = epochs
        elif capture_every == 'step':
            steps = epochs * B

        save_output = SaveOutput(N=N,
                                 steps=steps,
                                 attn_shape=attn_shape,
                                 chunks_shape=chunks_shape)
        hook_handles = []

        # hook model MultiheadAttention layer
        for layer in model.model.modules():
            if isinstance(layer, nn.modules.activation.MultiheadAttention):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

        if ON_CLUSTER:
            try:
                ret = subprocess.run([
                        'mail', '-s', f'"Status update {time():0.2f}"',
                        f'USER_EMAIL',
                        ],
                    input=f'Starting training... {epochs} {capture_every}s in total'.encode(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                    )
                print(f'sending mail: return {ret}')
            except Exception:
                traceback.print_exc()
        model.fit(epochs=epochs, losscurve=False)
        print('finished fitting, processing data')

        # all outputs for all layers N
        mod_outs = save_output.outputs


# %%
######################################################################
######################################################################
        for L in range(crabnet_params['N']):
            # one layer N (here we use layer zero)
            mod_out = mod_outs[L]

            n_mats = len(mod_out)

            mat_dims = np.array([mat.shape[0] for mat in mod_out])

            n_train = len(model.train_loader.dataset)
            n_val = len(model.data_loader.dataset)
            n_total = n_train + n_val

            n_mats = len(mod_out)  # number of output matrices from hook
            bsz = model.data_loader.batch_size  # batch size from data loader

            B_train = len(model.train_loader)  # total number of batches from train data loader
            B_val = len(model.data_loader)  # total number of batches from val data loader
            B = B_train + B_val  # total number of batches from data loader

            # Collect batches from validation set
            B = B_val

            # number of minibatches in epoch
            num_mbs_in_epoch = len(model.data_loader)

            H = model.model.heads  # number of heads
            N = model.model.N  # number of layers

            # n_data = len(model.data_loader.dataset)
            # n_data = n_total
            n_data = n_val
            # n_data = n_train
            n_elements = model.n_elements
######################################################################
######################################################################


# %%
        data_loader = model.data_loader

        form_out = pd.DataFrame(data=save_output.formulae)
        #discard unnecessary
        form_out = form_out[::N].reset_index(drop=True)

        # get final number of epochs from model
        if capture_every == 'epoch':
            epochs = model.epoch + 1
        elif capture_every == 'step':
            epochs = form_out.shape[0] // B

        print(f'epochs at the end: {epochs}')
        form_data = np.full(shape=(n_data, epochs), fill_value=np.nan, dtype=object)
        i = 0
        for mb in range(0, form_out.shape[0], B):
            # mb is minibatch
            mats = [np.array(form_out.loc[row]) for row in range(mb, mb + B)]
            mats = np.concatenate(mats)
            mats = mats[mats != None]
            form_data[:, i] = mats
            i += 1

        act_out = save_output.acts
        pred_out = save_output.preds

        act_out = act_out[::N*B]
        pred_out = pred_out[::N*B]

        act_data = np.full(shape=(n_data, epochs), fill_value=np.nan, dtype=np.float32)
        pred_data = np.full(shape=(n_data, epochs), fill_value=np.nan, dtype=np.float32)

        for mb in range(len(act_out)):
            act_data[:, mb] = np.array(act_out[mb])

        for mb in range(len(pred_out)):
            pred_data[:, mb] = np.array(pred_out[mb])

        print('Saving other data now')
        torch.save(data_loader, f'{data_save_path}/data_loader.pth')
        np.savez_compressed(f'{data_save_path}/act_data.npz', act_data=act_data)
        np.savez_compressed(f'{data_save_path}/pred_data.npz', pred_data=pred_data)
        np.savez_compressed(f'{data_save_path}/form_data.npz', form_data=form_data)

        tf = int(round(time() - t0_mp, ndigits=0))
        dt = str(timedelta(seconds=tf))
        print(f'training time for {mat_prop}: {dt}')


        # TESTING IF THE ARRAY STORE WORKED
        stores = [zarr.open_group(f'{data_save_path}/attn_data_layer{L}.zarr',
                                  mode='r') for L in range(3)]
        array_data = [stores[L][f'layer{L}'] for L in range(3)]
        print(array_data[0][0, 0, 0])


        print('now transforming Zarr arrays into ZIP arrays')
        ti = time()
        # now transforming Zarr arrays into ZIP arrays
        stores = [zarr.open_group(f'{data_save_path}/attn_data_layer{L}.zarr',
                          mode='r') for L in range(crabnet_params['N'])]
        array_data = [stores[L][f'layer{L}'] for L in range(crabnet_params['N'])]

        for layer in tqdm(range(len(array_data)), desc='processing layer'):
            attn_shape = array_data[layer].shape
            assert len(attn_shape) == 5

            total_size = attn_shape[0]
            n_data = act_data.shape[0]

            n_repeats = total_size // n_data
            assert n_data * n_repeats == total_size

            with zarr.ZipStore(f'{data_save_path}/attn_data_layer{layer}.zip',
                               mode='w') as store:
                root = zarr.group(store)

                chunks_shape = list(attn_shape)
                chunks_shape[0] = n_repeats
                chunks_shape[2] = 1
                chunks_shape = tuple(chunks_shape)
                print(f'chunks_shape: {chunks_shape}')

                attn0 = root.full(f'layer{layer}',
                                  shape=(attn_shape),
                                  chunks=chunks_shape,
                                  fill_value='NaN',
                                  dtype='float32',
                                  compression='blosc',
                                  compression_opts=dict(cname='lz4', clevel=5, shuffle=0),
                                  overwrite=True
                                  )

                for mb in tqdm(range(n_data), desc='processing chunks'):
                    idxs2 = slice(mb, attn_shape[0], n_data)
                    mats = array_data[layer][idxs2]
                    attn0[mb*n_repeats:(mb+1)*n_repeats, :, :, :, :] = mats

        tf = int(round(time() - t0_mp, ndigits=0))
        dt = str(timedelta(seconds=tf))
        print(f'transforming Zarr -> zipped Zarr time for {mat_prop}: {dt}')

        del_files = glob.glob(f'{data_save_path}/*.zarr')
        print('transforming to ZIP finished, deleting original zarr directories')
        for file in del_files:
            dest = shutil.rmtree(file)

        if ON_CLUSTER:
            print('Running on CLUSTER!')
            print(f'Moving all files from {data_save_path} to {data_save_path_orig}')
            dest = shutil.copytree(data_save_path, data_save_path_orig, dirs_exist_ok=True)
            print('Moving files seems to have worked... deleting the original files')
            dest = shutil.rmtree(data_save_path, ignore_errors=False)

    tf = int(round(time() - t0_all, ndigits=0))
    dt = str(timedelta(seconds=tf))
    print('script finished')
    print(f'total time elapsed: {dt}')
