import os

import numpy as np
import pandas as pd

import torch

import crabnet.kingcrab

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

from sklearn.metrics import roc_auc_score

from utils.utils import CONSTANTS, linear

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
data_dir = 'data/benchmark_data'
data_dir = 'data/benchmark_data'
mat_props = os.listdir(data_dir)

output_dir = 'data/embeddings_crabnet_onehot'
os.makedirs(output_dir, exist_ok=True)


# %%
def get_model(mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = rf'data/benchmark_data/{mat_prop}/train.csv'
    val_data = rf'data/benchmark_data/{mat_prop}/val.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_benchmark__predictions'
    # save_path = 'publication_predictions/onehot_benchmark__predictions'
    # save_path = 'publication_predictions/random_200_benchmark__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(f'{mat_prop}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = rf'data/benchmark_data/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(mat_prop, classification, file_name, verbose=True):
    model = load_model(mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'\n{mat_prop} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'\n{mat_prop} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)
    return model, mae


# %%
class SaveFormulae:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae_idx = []
        self.x_emb = []
        self.acts = []
        self.preds = []
        self.counter = 0

    def __call__(self, module, module_in, module_out):
        # only capture output if requires_grad == False (i.e. in validation)
        # only capture output if requires_grad == True (i.e. in training)
        if model.capture_flag:
            src = module_in[0]
            # indices representing atoms for each compound
            self.formulae_idx.append(src)
            # embedded atoms for each compound
            self.x_emb.append(module.embed(src) * 2**module.emb_scaler)
            # the module_out here is the EDM' tensor ((Q*KT)*V)
            self.outputs.append(module_out)
            self.counter += 1

    def clear(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae_idx = []
        self.acts = []
        self.preds = []
        self.counter = []


class SaveQKV:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = 0

    def __call__(self, module, module_in, module_out):
        # only capture output if requires_grad == False (i.e. in validation)
        # only capture output if requires_grad == True (i.e. in training)
        if model.capture_flag == True:
            # module_inputs are the "x" tensors
            self.inputs.append(*module_in)
            # weights are the attention in projection weights
            self.weights.append(module.self_attn.in_proj_weight)
            # biases are the attention in projection biases
            self.biases.append(module.self_attn.in_proj_bias)
            # plain-text full formula strings
            self.formulae.append(model.formula_current)
            # self.acts.append(model.act_v)
            # self.preds.append(model.pred_v)
            self.counter += 1

    def clear(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.formulae = []
        self.acts = []
        self.preds = []
        self.counter = []


# %%
cons = CONSTANTS()
elements = cons.atomic_symbols


# %%
for mat_prop in mat_props:
    print(f'{mat_prop= }')

    # instantiate CrabNet model that doesn't drop unary (pure) compounds
    model = Model(CrabNet(compute_device=compute_device,
                          residual_nn='roost')
                  .to(compute_device),
                  model_name=f'{mat_prop}',
                  verbose=True,
                  drop_unary=False,
                  scale=False)
    model.load_network(f'{mat_prop}.pth')

    # Manually replace dataset with all elements
    data = 'data/all_elements.csv'

    model.load_data(data, batch_size=2**9, train=False)

    # manually set all EDM fractional amounts to be 0
    # model.data_loader.dataset.data[0][:,1,:] = 1.0
    # model.data_loader.dataset.data[2] = np.array(elements[1:])

    # set model capture flag to true
    model.capture_flag = True
    mod_list = [m for m in model.model.modules()]

    # %%
    save_formulae = SaveFormulae()
    hook_handles_formulae = []

    for layer in model.model.modules():
        # if isinstance(layer, torch.nn.TransformerEncoderLayer):
        if isinstance(layer, crabnet.kingcrab.Encoder):
            print('hooked formulae')
            handle = layer.register_forward_hook(save_formulae)
            hook_handles_formulae.append(handle)

    # %%
    save_qkv = SaveQKV()
    hook_handles_qkv = []

    for layer in model.model.modules():
        if isinstance(layer, torch.nn.TransformerEncoderLayer):
            print('hooked qkv')
            handle = layer.register_forward_hook(save_qkv)
            hook_handles_qkv.append(handle)

    # %% run model to populate hooks
    model, output = get_results(model)

    # %% work with the weights
    n_hooks = len(hook_handles_qkv)

    for layer in [0, 1, 2]:
        # inputs has the dimension [n_el, bsz, d_model]
        n_el, bsz, d_model = save_qkv.inputs[layer].shape

        weights = save_qkv.weights[layer]
        biases = save_qkv.biases[layer]
        querys = save_qkv.inputs[layer::n_hooks]
        querys = torch.cat(querys, dim=1)

        full_formulae = save_qkv.formulae[0]


        # %%
        Q, K, V = linear(querys, weights, biases).chunk(3, dim=-1)

        Q = Q.detach().cpu().numpy()
        K = K.detach().cpu().numpy()
        V = V.detach().cpu().numpy()

        Q = Q.squeeze(0)
        K = K.squeeze(0)
        V = V.squeeze(0)


        # %%
        # export embeddings to csv
        dfq = pd.DataFrame(Q, index=elements[1:])
        dfk = pd.DataFrame(K, index=elements[1:])
        dfv = pd.DataFrame(V, index=elements[1:])

        dfq.to_csv(f'{output_dir}/{mat_prop}_layer{layer}_Q.csv')
        dfk.to_csv(f'{output_dir}/{mat_prop}_layer{layer}_K.csv')
        dfv.to_csv(f'{output_dir}/{mat_prop}_layer{layer}_V.csv')

