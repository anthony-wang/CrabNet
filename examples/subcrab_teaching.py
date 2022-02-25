import torch
from torch import nn
from crabnet.kingcrab import ResidualNetwork, Embedder, FractionalEncoder


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

        self.out_dims = out_dims
        self.d_model = d_model

        # embed the elemental features
        self.embed = Embedder(
            d_model=d_model, compute_device=compute_device, elem_prop=elem_prop
        )

        # encode "positions" of fractional contributions
        self.pe = FractionalEncoder(d_model, log10=False)
        self.ple = FractionalEncoder(d_model, log10=True)

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # residual network
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
        fc_elem_emb = self.embed(src)
        # # mask has 1 if n-th element is present, 0 if not. E.g. single element compound has mostly mask of 0's
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        prevalence_encodings = torch.zeros_like(fc_elem_emb)
        prevalence_log_encodings = torch.zeros_like(fc_elem_emb)

        # fractional encoding, see Fig 6 of 10.1038/s41524-021-00545-1
        # first half of features are prevalence encoded (i.e. 512//2==256)
        prevalence_encodings[:, :, : self.d_model // 2] = self.pe(frac)
        # second half of features are prevalence log encoded
        prevalence_log_encodings[:, :, self.d_model // 2 :] = self.ple(frac)

        # sum of fc_mat2vec embedding (x), prevalence encoding (pe), and prevalence log
        # encoding (ple), see Fig 6 of 10.1038/s41524-021-00545-1 (ple not shown)
        x_src = fc_elem_emb + prevalence_encodings + prevalence_log_encodings
        x_src = x_src.transpose(0, 1)

        # transformer encoding
        # True (1) values in `src_mask` mean ignore the corresponding value in the
        # attention layer. Source:
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # See also
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        x = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
        x = x.transpose(0, 1)

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
