import shutil
import os


# %%
def copy_weights(source, dest, seed):
    dir_list = os.listdir(source)
    seed = f'seed{seed}'

    dir_list_seed = [dlist for dlist in dir_list if seed in dlist]

    for d in dir_list_seed:
        in_path = f'{source}/{d}/checkpoint.pth'
        network = d.split(f'{seed}-')[-1].split('-element')[0]
        prop = d.split(f'{seed}-')[-1].split('-element')[-1]
        out_path = f'{dest}/{network}{prop}.pth'

        shutil.copy2(in_path, out_path)


# %%
if __name__ == '__main__':
    path = r'data/score_summaries/NN/Run_publication'

    dest_path = r'data/user_properties/trained_weights'
    os.makedirs(dest_path, exist_ok=True)

    RNG_SEED = 9

    copy_weights(path, dest_path, RNG_SEED)
