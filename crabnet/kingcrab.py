"""Contains classes for transformer architecture within CrabNet."""
from os.path import join, dirname

import numpy as np
import pandas as pd

import torch
from torch import nn
from collections import OrderedDict

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.

    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, bias=False):
        """Instantiate a ResidualNetwork model.

        Parameters
        ----------
        input_dim : int
            Input dimensions for the Residual Network, specified in SubCrab() model class, by default 512
        output_dim : int
            Output dimensions for Residual Network, by default 3
        hidden_layer_dims : list(int)
            Hidden layer architecture for the Residual Network, by default [1024, 512, 256, 128]
        bias : bool
            Whether to bias the linear network, by default False
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=bias)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        """Propagate Residual Network weights forward.

        Parameters
        ----------
        fea : torch.tensor (n_dim)
            Tensor output of self attention block
        Returns
        -------
        fc_out
            The output of the Residual Network
        """
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        """Return the class name."""
        return f"{self.__class__.__name__}"


class TransferNetwork(nn.Module):
    """Learn extended representations of materials during transfer learning.

    This network was designed to have little impact on predictions during
    training and enhance learning with the inclusion of extended features.
    """

    def __init__(self, input_dims, output_dims):
        """Instantiate a TransferNetwork to learn extended representations.

        Parameters
        ----------
        input_dims : int
            Dimensions of input layer

        output_dims : int
            Dimensions of output layer
        """
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dims, 512)),
                    ("leakyrelu1", nn.LeakyReLU()),
                    ("fc2", nn.Linear(512, output_dims)),
                    ("leakyrelu2", nn.LeakyReLU()),
                ]
            )
        )

    def forward(self, x):
        """Perform a forward pass of the TransferNetwork.

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = self.layers(x)
        return x


class Embedder(nn.Module):
    """Perform composition-based embeddings of elemental features."""

    def __init__(
        self,
        d_model: int,
        compute_device: str = None,
        elem_prop: str = "mat2vec",
    ):
        """Embed elemental features, similar to CBFV.

        Parameters
        ----------
        d_model : int
            Row dimenions of elemental emeddings, by default 512
        compute_device : str
            Name of device which the model will be run on
        elem_prop : str
            Which elemental feature vector to use. Possible values are "jarvis",
            "magpie", "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by
            default "mat2vec"
        """
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = join(dirname(__file__), "data", "element_properties")
        # # Choose what element information the model receives
        mat2vec = join(elem_dir, elem_prop + ".csv")  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        # NOTE: Parameters within nn.Embedding
        self.cbfv = nn.Embedding.from_pretrained(cat_array).to(
            self.compute_device, dtype=data_type_torch
        )

    def forward(self, src):
        """Compute forward call for embedder class to perform elemental embeddings.

        Parameters
        ----------
        src : torch.tensor
            Tensor containing element numbers corresponding to elements in compound

        Returns
        -------
        torch.tensor
            Tensor containing elemental embeddings for compounds, reduced to d_model dimensions
        """
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


# %%
class FractionalEncoder(nn.Module):
    """Encode element fractional amount using a "fractional encoding".

    This is inspired by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, resolution=100, log10=False, compute_device=None):
        """Instantiate the FractionalEncoder.

        Parameters
        ----------
        d_model : int
            Model size, see paper, by default 512
        resolution : int
            Number of discretizations for the fractional prevalence encoding, by default 100
        log10 : bool
            Whether to apply a log operation to fraction prevalence encoding, by default False
        compute_device : str
            The compute device to store and run the FractionalEncoder class
        """
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(
            0, self.resolution - 1, self.resolution, requires_grad=False
        ).view(self.resolution, 1)
        fraction = (
            torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False)
            .view(1, self.d_model)
            .repeat(self.resolution, 1)
        )

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer("pe", pe)

    def forward(self, x):
        """Perform the forward pass of the fractional encoding.

        Parameters
        ----------
        x : torch.tensor
            Tensor of linear spaced values based on defined resolution

        Returns
        -------
        out
            Sinusoidal expansions of elemental fractions
        """
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
        x[x < 1 / self.resolution] = 1 / self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out


# %%
class Encoder(nn.Module):
    """Create elemental descriptor matrix via element embeddings and frac. encodings.

    See the CrabNet paper for further details:
    https://www.nature.com/articles/s41524-021-00545-1
    """

    def __init__(
        self,
        d_model,
        N,
        heads,
        extend_features=None,
        fractional=True,
        attention=True,
        compute_device=None,
        pe_resolution=5000,
        ple_resolution=5000,
        elem_prop="mat2vec",
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        """Instantiate the Encoder class to create elemental descriptor matrix (EDM).

        Parameters
        ----------
        d_model : _type_
            _description_
        N : int, optional
            Number of encoder layers, by default 3
        heads : int, optional
            Number of attention heads to use, by default 4
        extend_features : Optional[List[str]]
            Additional features to grab from columns of the other DataFrames (e.g. state
            variables such as temperature or applied load), by default None
        fractional : bool, optional
            Whether to weight each element by its fractional contribution, by default True.
        attention : bool, optional
            Whether to perform self attention, by default True
        pe_resolution : int, optional
            Number of discretizations for the prevalence encoding, by default 5000
        ple_resolution : int, optional
            Number of discretizations for the prevalence log encoding, by default 5000
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis",
            "magpie", "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by
            default "mat2vec"
        emb_scaler : float, optional
            _description_, by default 1.0
        pos_scaler : float, optional
            Scaling factor applied to fractional encoder, by default 1.0
        pos_scaler_log : float, optional
            Scaling factor applied to log fractional encoder, by default 1.0
        dim_feedforward : int, optional
            Dimenions of the feed forward network following transformer, by default 2048
        dropout : float, optional
            Percent dropout in the feed forward network following the transformer, by default 0.1
        """
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.extend_features = extend_features
        self.fractional = fractional
        self.attention = attention
        self.compute_device = compute_device
        self.pe_resolution = pe_resolution
        self.ple_resolution = ple_resolution
        self.elem_prop = elem_prop
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)
        self.prevalence_encoder = FractionalEncoder(
            self.d_model, resolution=pe_resolution, log10=False
        )
        self.prevalence_log_encoder = FractionalEncoder(
            self.d_model, resolution=ple_resolution, log10=True
        )

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([emb_scaler]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([pos_scaler]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([pos_scaler_log]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                self.d_model,
                nhead=self.heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.N
            )

    def forward(self, src, frac, extra_features=None):
        """Compute the forward pass for encoding the elemental descriptor matrix.

        Parameters
        ----------
        src : torch.tensor
            Tensor containing integers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing the fractions of each element in compound
        extra_features : bool, optional
            Whether to append extra features after encoding, by default None

        Returns
        -------
        torch.tensor
            Tensor containing flattened transformer representations of compounds
            concatenated with extended features.
        """
        x = self.embed(src) * self.emb_scaler  # * 2 ** self.emb_scaler

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = self.pos_scaler
        ple_scaler = self.pos_scaler_log
        pe[:, :, : self.d_model // 2] = self.prevalence_encoder(frac) * pe_scaler
        ple[:, :, self.d_model // 2 :] = self.prevalence_log_encoder(frac) * ple_scaler

        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        if self.attention:
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        if self.extend_features is not None:
            n_elements = x.shape[1]
            X_extra = extra_features.repeat(1, 1, n_elements).permute([1, 2, 0])
            x = torch.concat((x, X_extra), axis=2)

        return x


# %%
class SubCrab(nn.Module):
    """SubCrab model class which implements the transformer architecture."""

    def __init__(
        self,
        out_dims=3,
        d_model=512,
        extend_features=None,
        d_extend=0,
        N=3,
        heads=4,
        fractional=False,
        attention=True,
        compute_device=None,
        out_hidden=[1024, 512, 256, 128],
        pe_resolution=5000,
        ple_resolution=5000,
        elem_prop="mat2vec",
        bias=False,
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        """Instantiate a SubCrab class to be used within CrabNet.

        Parameters
        ----------
        out_dims : int, optional
            Output dimensions for Residual Network, by default 3
        d_model : int, optional
            Model size. See paper, by default 512
        extend_features : _type_, optional
            Additional features to grab from columns of the other DataFrames (e.g. state
            variables such as temperature or applied load), by default None
        d_extend : int, optional
            Number of extended features, by default 0
        N : int, optional
            Number of attention layers, by default 3
        heads : int, optional
            Number of attention heads, by default 4
        frac : bool, optional
            Whether to multiply `x` by the fractional amounts for each element, by default False
        attn : bool, optional
            Whether to perform self attention, by default True
        compute_device : _type_, optional
            Computing device to run model on, by default None
        out_hidden : list(int), optional
            Architecture of hidden layers in the Residual Network, by default [1024, 512, 256, 128]
        pe_resolution : int, optional
            Number of discretizations for the prevalence encoding, by default 5000
        ple_resolution : int, optional
            Number of discretizations for the prevalence log encoding, by default 5000
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
        bias : bool, optional
            Whether to bias the Residual Network, by default False
        emb_scaler : float, optional
            Float value by which to scale the elemental embeddings, by default 1.0
        pos_scaler : float, optional
            Float value by which to scale the fractional encodings, by default 1.0
        pos_scaler_log : float, optional
            Float value by which to scale the log fractional encodings, by default 1.0
        dim_feedforward : int, optional
            Dimenions of the feed forward network following transformer, by default 2048
        dropout : float, optional
            Percent dropout in the feed forward network following the transformer, by default 0.1
        """
        super().__init__()
        self.avg = True
        self.out_dims = out_dims
        self.d_model = d_model
        self.extend_features = extend_features
        self.d_extend = d_extend
        self.N = N
        self.heads = heads
        self.fractional = fractional
        self.attention = attention
        self.compute_device = compute_device
        self.bias = bias
        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            heads=self.heads,
            attention=self.attention,
            compute_device=self.compute_device,
            pe_resolution=pe_resolution,
            ple_resolution=ple_resolution,
            elem_prop=elem_prop,
            emb_scaler=emb_scaler,
            pos_scaler=pos_scaler,
            pos_scaler_log=pos_scaler_log,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.out_hidden = out_hidden
        self.output_nn = ResidualNetwork(
            self.d_model + self.d_extend,
            self.out_dims,
            self.out_hidden,
            self.bias,
        )

    def forward(self, src, frac, extra_features=None):
        """Compute forward pass of the SubCrab model class (i.e. transformer).

        Parameters
        ----------
        src : torch.tensor
            Tensor containing element numbers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing fractional amounts of each element in compound
        extra_features : bool, optional
            Whether to append extra features after encoding, by default None

        Returns
        -------
        torch.tensor
            Model output containing predicted value and uncertainty for that value
        """
        output = self.encoder(src, frac, extra_features)
        # output = self.transfer_nn(output)

        # average the "element contribution", mask so you only average "elements" (i.e.
        # not padded zero values)
        elem_pad_mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)  # simple linear
        if self.avg:
            output = output.masked_fill(elem_pad_mask, 0)
            output = output.sum(dim=1) / (~elem_pad_mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability

        return output


# %%
if __name__ == "__main__":
    model = SubCrab()
