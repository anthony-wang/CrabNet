"""Contains classes for transformer architecture within CrabNet."""
from os.path import join, dirname

import numpy as np
import pandas as pd

import torch
from torch import nn

data_type_torch = torch.float32

# %%
class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.

    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """Instantiate a ResidualNetwork model.

        Parameters
        ----------
        input_dim : int
            Input dimensions for the Residual Network, specified in SubCrab() model class, by default 512
        output_dim : int
            Output dimensions for Residual Network, by default 3
        hidden_layer_dims : list(int)
            Hidden layer architecture for the Residual Network, by default [1024, 512, 256, 128]
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
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
        elem_vec = join(elem_dir, elem_prop + ".csv")  # element embedding

        cbfv = pd.read_csv(elem_vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)

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
class SubCrab(nn.Module):
    """SubCrab model class which implements the transformer architecture."""

    def __init__(
        self,
        out_dims=3,
        d_model=512,
        heads=4,
        compute_device=None,
        elem_prop="mat2vec",
    ):
        """Instantiate a SubCrab class to be used within CrabNet.

        Parameters
        ----------
        out_dims : int, optional
            Output dimensions for Residual Network, by default 3
        d_model : int, optional
            Model size. See paper, by default 512
        compute_device : _type_, optional
            Computing device to run model on, by default None
        elem_prop : str, optional
            Which elemental feature vector to use. Possible values are "jarvis", "magpie",
            "mat2vec", "oliynyk", "onehot", "ptable", and "random_200", by default "mat2vec"
        """
        super().__init__()
        self.avg = True
        self.out_dims = 3
        self.compute_device = compute_device

        # %% Encoder
        self.fractional = False
        self.attention = True
        self.elem_prop = elem_prop
        self.embed = Embedder(d_model=self.d_model, compute_device=self.compute_device)

        self.pe = FractionalEncoder(self.d_model, log10=False)
        self.ple = FractionalEncoder(self.d_model, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.0]))

        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=heads)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=3
            )

        self.output_nn = ResidualNetwork(d_model, out_dims, [1024, 512, 256, 128])

    def forward(self, src, frac):
        """Compute forward pass of the SubCrab model class (i.e. transformer).

        Parameters
        ----------
        src : torch.tensor
            Tensor containing element numbers corresponding to elements in compound
        frac : torch.tensor
            Tensor containing fractional amounts of each element in compound

        Returns
        -------
        torch.tensor
            Model output containing predicted value and uncertainty for that value
        """
        # %% Encoder
        x = self.embed(src) * self.emb_scaler
        # # mask has 1 if n-th element is present, 0 if not. E.g. single element compound has mostly mask of 0's
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)

        # fractional encoding, see Fig 6 of 10.1038/s41524-021-00545-1
        pe_scaler = self.pos_scaler
        ple_scaler = self.pos_scaler_log
        # first half of features are prevalence encoded (i.e. 512//2==256)
        pe[:, :, : self.d_model // 2] = self.pe(frac) * pe_scaler
        # second half of features are prevalence log encoded
        ple[:, :, self.d_model // 2 :] = self.ple(frac) * ple_scaler

        if self.attention:
            # sum of fc_mat2vec embedding (x), prevalence encoding (pe), and prevalence log encoding (ple)
            # see Fig 6 of 10.1038/s41524-021-00545-1
            x_src = x + pe + ple
            x_src = x_src.transpose(0, 1)

            # transformer encoding
            # True (1) values in `src_mask` mean ignore the corresponding value in the
            # attention layer. Source:
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            # See also
            # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
            x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        # 0:1 index eliminates the repeated values (down to 1 colummn) repeat() fills it back up (to e.g. d_model == 512 values)
        hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            # set values of x which correspond to an element not being present to 0
            x = x.masked_fill(hmask == 0, 0)

        # average the "element contribution" at the end, mask so you only average "elements"
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        x = self.output_nn(x)  # simple linear

        # average the attention heads
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=1) / (~mask).sum(dim=1)
        x, logits = x.chunk(2, dim=-1)
        probability = torch.ones_like(x)
        probability[:, : logits.shape[-1]] = torch.sigmoid(logits)
        x = x * probability

        return x
