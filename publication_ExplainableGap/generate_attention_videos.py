import os
from copy import deepcopy
import shutil
import glob
import re

import subprocess
import traceback

import sys
import multiprocessing as mp

import numpy as np
import torch

import zarr

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from time import time
from datetime import timedelta
from tqdm import tqdm

from itertools import repeat

from utils.get_compute_device import get_compute_device
from utils.get_core_count import get_core_count
from utils.attention_utils import get_form, plot_progress, plot_all_heads,\
    plot_all_heads_save, plot_progress_save, imgs_to_video, vids_to_video


# %%
# find specific elements:
idxs = []

# idxs for aflow_bulk_mod
idxs.append(75)  # P1W3
idxs.append(25)  # C11N4
idxs.append(64)  # Hf1O2
idxs.append(93)  # Pt3Ti1
idxs.append(98)  # Ga1N1
idxs.append(135)  # Gd2O7Ti2
idxs.append(279)  # B1N1
idxs.append(348)  # Al1Gd1O3
idxs.append(627)  # Hf2La2O7

# idxs for mp_bulk_modulus
idxs.append(29)  # SiO2
idxs.append(68)  # FeCO3
idxs.append(66)  # YSF
idxs.append(118)  # Fe(HO)2
idxs.append(124)  # AgSbTe2
idxs.append(221)  # NaNO3
idxs.append(304)  # Al2O3
idxs.append(616)  # LiFeO2

# NOTE: this takes a long time, for demo purposes, select fewer compounds
idxs = idxs[0:3]


# %%
if __name__ == '__main__':
    ON_CLUSTER = os.environ.get('ON_CLUSTER')
    HOME = os.environ.get('HOME')
    USER_EMAIL = os.environ.get('USER_EMAIL')
    FAST_DRIVE = os.environ.get('FAST_DRIVE')
    if ON_CLUSTER:
        print('Running on CLUSTER!')
    if FAST_DRIVE:
        print('This system has a fast drive!')

    print('Generating attention/progress plots and videos')
    compute_device = get_compute_device(prefer_last=True)
    cuda_avail = torch.cuda.is_available()

    if mp.get_start_method(allow_none=True) is None:
        if sys.platform == 'win32':
            mp.set_start_method('spawn')
        elif sys.platform == 'linux':
            mp.set_start_method('fork')
        elif sys.platform == 'darwin':
            mp.set_start_method('fork')

    print(f'mp start method: {mp.get_start_method(allow_none=False)}')

    n_cores = int(get_core_count())
    max_procs = 16
    print(f'detected number of cores: {n_cores}')
    n_cores = min(n_cores, max_procs)
    print(f'parallel processes to spawn: {n_cores} (max_cores: {max_procs})')

    data_save_path_base_orig = 'data_save/'
    data_save_path_base = deepcopy(data_save_path_base_orig)

    if ON_CLUSTER:
        prepend_path = ''
        data_save_path_base = prepend_path + data_save_path_base_orig

    print(f'data_save_path_base: {data_save_path_base}')
    mat_props = os.listdir(data_save_path_base)
    mat_props = [m for m in mat_props if '.DS_Store' not in m]
    mat_props = ['aflow__Egap']


    # extract FFmpeg version and determine encoder to use
    FFMPEG_VERSION = None
    encoder = 'mpeg4' # default (fallback) encoder to use (low quality)
    try:
        ret = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE)
        ret = ret.stdout.decode('utf-8')
        regex = r'(?:ffmpeg version)\s([\d]*\.[\d]*)'
        regex = re.compile(regex, re.IGNORECASE)
        match = re.findall(regex, ret)
        ffmpeg_version = np.float32(match[-1])
        FFMPEG_VERSION = np.floor(ffmpeg_version)
        if ON_CLUSTER:
            encoder = 'libx264' # CPU encoder, relatively good and fast
        elif torch.cuda.is_available():
            encoder = 'h264_nvenc' # NVIDIA encoder, fast and good quality
        elif FFMPEG_VERSION >= 4.0:
            # HACK: this assumes that FFmpeg versions > 4.0 have libx264
            encoder = 'libx264' # CPU encoder, relatively good and fast
    except Exception:
        traceback.print_exc()
        print('ERROR in FFmpeg version detection, will use fallback '
              f'encoder {encoder}')
    print(f'FFmpeg base version: {FFMPEG_VERSION}')
    print(f'FFmpeg encoder to be used: {encoder}')

    print(f'mat_props to process: #{len(mat_props)}, {mat_props}')
    print(f'compositions to process: #{len(idxs)}')

    t0_all = time()

    for mat_prop in tqdm(mat_props, desc='processing mat_prop'):
        data_save_path = f'{data_save_path_base}/{mat_prop}'
        try:
            data_loader = torch.load(f'{data_save_path}/data_loader.pth')
            with np.load(f'{data_save_path}/act_data.npz') as data:
                act_data = data['act_data']
            with np.load(f'{data_save_path}/pred_data.npz') as data:
                pred_data = data['pred_data']
            with np.load(f'{data_save_path}/form_data.npz', allow_pickle=True) as data:
                form_data = data['form_data']

            # zip zarr arrays
            attn_files = [f'{data_save_path}/attn_data_layer{L}.zip' for L in range(3)]
            attn_data = [zarr.open(file, mode='r') for file in attn_files]
            attn_data = [group[f'layer{L}'] for L, group in enumerate(attn_data)]

        except Exception:
            print('LOADING PROBLEM!')
            traceback.print_exc()
            continue

        out_dir_base_orig = 'figures/paying_attention'
        out_dir_orig = f'{out_dir_base_orig}/{mat_prop}'
        out_dir_base = deepcopy(out_dir_base_orig)
        out_dir = deepcopy(out_dir_orig)

        if ON_CLUSTER:
            if cuda_avail:
                device_name = torch.cuda.get_device_name(0)
                if isinstance(device_name, str) and ('RTX' in device_name):
                    prepend_path = '/scratch/awang/'
            elif FAST_DRIVE:
                prepend_path = '/scratch/awang/'
            else:
                prepend_path = '/fast/awang/'
            out_dir_base = prepend_path + out_dir_base_orig
            out_dir = prepend_path + out_dir_orig
            print(f'{out_dir_base_orig = }')
            print(f'{out_dir_orig = }')
            print(f'{out_dir_base = }')
            print(f'{out_dir = }')

        os.makedirs(out_dir_base_orig, exist_ok=True)
        os.makedirs(out_dir_orig, exist_ok=True)
        os.makedirs(out_dir_base, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        if ON_CLUSTER:
            try:
                ret = subprocess.run([
                        'mail', '-s',
                        f'"Status update {time():0.2f}"',
                        f'USER_EMAIL',
                        ],
                    input='Training is done, now processing images / videos'.encode(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    )
                print(f'sending mail: return {ret}')
            except Exception:
                traceback.print_exc()

        t0_mp = time()
        layer = 0
        epochs = act_data.shape[-1]

        n_digits = len(str(epochs))
        n_plots = epochs

        # new stride (for Zarr arrays)
        stride = attn_data[0].chunks[0]

        chunksize = int(np.ceil(n_plots/n_cores))
        print(f'chunksize: {chunksize}')
        assert chunksize > 0

        for idx in tqdm(idxs, desc='processing composition'):
            if attn_data is None:
                print('attn_data IS NONE!? WHY?')
                continue
            print(f'currently on idx: {idx}, formula: {form_data[idx, 0]}')
            t0_idx = time()

            form = get_form(data_loader, idx=idx)

            err_msg = 'input attn_data is of the wrong shape'
            assert len(attn_data[0].shape) == 5, err_msg
            # ##############################################
            # ##############################################
            # mp plot attention
            # generate sacrificial plots and object handles
            a, b, c, d = plot_all_heads(data_loader, attn_data, stride,
                                        idx=idx, epoch=0, layer=layer, mask=True)

            fignames = ['{}/{}_step{}.png'.format(out_dir, form, f'{n:0{n_digits}d}')
                        for n in range(n_plots)]
            args1 = zip(n_plots*[data_loader], repeat(attn_data), repeat(stride),
                       repeat(a), repeat(b), repeat(c), repeat(d),
                       repeat(idx), range(epochs),
                       repeat(layer), repeat(True), fignames)

            ti = time()
            pool1 = mp.Pool(n_cores)
            result = pool1.starmap_async(plot_all_heads_save, args1,
                                         chunksize=chunksize)
            pool1.close()
            pool1.join()
            dt = time() - ti
            plt.close('all')
            print(f'\nattention plots/s: {n_plots/dt:0.3f}')

            # mp plot progress
            print('... batch processing progress plots using multiprocessing')
            # generate sacrificial plots and object handles
            a, b, c, d, e = plot_progress(form, mat_prop, idx, act_data, pred_data,
                                          epoch=0, epochs=epochs)

            fignames = ['{}/{}_progress_step{}.png'
                        .format(out_dir, form,f'{n:0{n_digits}d}')
                        for n in range(n_plots)]
            args2 = zip(n_plots*[form], repeat(mat_prop),
                       repeat(a), repeat(b), repeat(c), repeat(d), repeat(e),
                       repeat(idx), repeat(act_data), repeat(pred_data),
                       range(epochs), repeat(epochs), fignames)

            ti = time()
            pool2 = mp.Pool(n_cores)
            result = pool2.starmap_async(plot_progress_save, args2,
                                         chunksize=chunksize)
            pool2.close()
            pool2.join()
            dt = time() - ti
            plt.close('all')
            print(f'\nprogress plots/s: {n_plots/dt:0.3f}')
            # ##############################################
            # ##############################################

            rets = []
            t1 = time()
            print('... generating videos of attention maps')
            # make H264 mp4 of attention maps
            ret = imgs_to_video(
                infile=f'{out_dir}/{form}_step%{n_digits}d.png',
                outfile=f'{out_dir}/{form}_fast.mp4',
                encoder=f'{encoder}'
                )
            rets.append(ret.returncode)
            if ret.returncode != 0:
                print(ret.args)
                if ret.stdout is not None:
                    print(ret.stdout.decode('utf-8'))
                if ret.stderr is not None:
                    print(ret.stderr.decode('utf-8'))
            dt1 = time() - t1

            t2 = time()
            print('... generating videos of progress plots')
            # make H264 mp4 of progress
            ret = imgs_to_video(
                infile=f'{out_dir}/{form}_progress_step%{n_digits}d.png',
                outfile=f'{out_dir}/{form}_progress_fast.mp4',
                encoder=f'{encoder}'
                )
            rets.append(ret.returncode)
            if ret.returncode != 0:
                print(ret.args)
                if ret.stdout is not None:
                    print(ret.stdout.decode('utf-8'))
                if ret.stderr is not None:
                    print(ret.stderr.decode('utf-8'))
            dt2 = time() - t2

            t3 = time()
            print('... merging videos')
            # merge attention map and progress videos
            ret = vids_to_video(
                infile1=f'{out_dir}/{form}_fast.mp4',
                infile2=f'{out_dir}/{form}_progress_fast.mp4',
                outfile=f'{out_dir}/{form}_merged_fast.mp4',
                encoder=f'{encoder}'
                )
            rets.append(ret.returncode)
            if ret.returncode != 0:
                print(ret.args)
                if ret.stdout is not None:
                    print(ret.stdout.decode('utf-8'))
                if ret.stderr is not None:
                    print(ret.stderr.decode('utf-8'))
            dt3 = time() - t3

            delete_files = True
            rets = np.array(rets)
            if len(rets) == 0:
                continue
            elif np.any(rets != 0):
                print(rets)
                print('ERROR: FFmpeg processes did NOT return all zeros!')
                break
            # if all processes return zero, delete png files
            elif rets[0] == 0 and np.sum(rets == rets[0]) == len(rets):
                del_files = glob.glob(f'{out_dir}/*.png')
                form_escaped = re.escape(f'{form}')
                regex = r'{}(_progress)?_step[0-9]{{{}}}.png'.format(form_escaped, n_digits)
                r = re.compile(regex)
                if delete_files:
                    print('--> all ffmpeg processes returned 0, deleting .png files')
                    for file in del_files:
                        basename = os.path.basename(file)
                        if re.search(r, basename):
                            os.remove(file)
            else:
                print('ERROR: FFmpeg processes did NOT return all zeros!')
                break

            print(f'... FFmpeg times: {dt1:0.3f}, {dt2:0.3f}, {dt3:0.3f}, '
                  f'total: {(dt1 + dt2 + dt3):0.3f}')

            tf = int(round(time() - t0_idx, ndigits=0))
            dt = str(timedelta(seconds=tf))
            print(f'... generation time for #{idx} {form}: {dt}')

        if ON_CLUSTER:
            print(f'Moving all files from {out_dir_base} to {out_dir_base_orig}')
            # copy all files from /fast/ to the working drive
            dest = shutil.copytree(out_dir_base, out_dir_base_orig, dirs_exist_ok=True)
            print('Copying files seems to have worked...')
            if delete_files:
                print('... deleting the original files now')
                dest = shutil.rmtree(out_dir_base, ignore_errors=False)

        tf = int(round(time() - t0_mp, ndigits=0))
        dt = str(timedelta(seconds=tf))
        print(f'generation time for {mat_prop}: {dt}')

    tf = int(round(time() - t0_all, ndigits=0))
    dt = str(timedelta(seconds=tf))
    print('script finished')
    print(f'TOTAL generation time: {dt}')
