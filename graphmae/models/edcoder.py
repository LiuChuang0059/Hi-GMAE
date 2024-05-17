from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gcn import GCN
from .gat import GAT
from .gin import GIN
from .loss_func import sce_loss
from graphmae.utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, remove_self_loops

from .transformer import Transformer


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop,  pe_dim, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    elif m_type == 'gcn':
        mod = GCN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == 'transformer':
        mod = Transformer(
            in_dim=int(in_dim),
            pe_dim=pe_dim,
            hidden_size=int(num_hidden) * nhead,
            ffn_size=int(out_dim) * nhead,
            dropout_rate=dropout,
            attention_dropout_rate=attn_drop,
            num_heads=nhead
        )
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            coarse_layer: int,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            pe_dim: int,
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            last_enc_type: str = "transformer",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._coarse_layer = coarse_layer
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.proj = nn.Linear(num_hidden, in_dim)

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoders = nn.ModuleList()
        for i in range(coarse_layer):
            if i != 0:
                enc_in_dim = num_hidden
                layer = 1
            else:
                enc_in_dim = in_dim
                layer = num_layers
            enc = encoder_type if i != coarse_layer - 1 else last_enc_type
            encoder = setup_module(
                m_type=enc,
                enc_dec="encoding",
                in_dim=enc_in_dim,
                num_hidden=enc_num_hidden,
                out_dim=enc_num_hidden,
                num_layers=layer,
                nhead=enc_nhead,
                nhead_out=enc_nhead,
                concat_out=True,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                pe_dim=pe_dim
            )
            self.encoders.append(encoder)

        # build decoder for attribute prediction
        self.decoders = nn.ModuleList()
        for i in range(coarse_layer):
            if i != 0:
                dec_out_dim = num_hidden
            else:
                dec_out_dim = in_dim
            decoder = setup_module(
                m_type=decoder_type,
                enc_dec="decoding",
                in_dim=num_hidden,
                num_hidden=dec_num_hidden,
                out_dim=dec_out_dim,
                num_layers=1,
                nhead=nhead,
                nhead_out=nhead_out,
                activation=activation,
                dropout=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm,
                concat_out=True,
                pe_dim=pe_dim
            )
            self.decoders.append(decoder)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(num_hidden * num_layers, num_hidden, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, num_nodes, device):
        mask_rate = self._mask_rate
        perm = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        # keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            # noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            # noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]
        else:
            token_nodes = mask_nodes
        """
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        """
        # out_x[token_nodes] += self.enc_mask_token
        # return out_x, (mask_nodes, token_nodes)
        mask_node = torch.ones(num_nodes, dtype=torch.float, device=device)
        mask_node[mask_nodes] = 0
        token_node = torch.ones(num_nodes, dtype=torch.float, device=device)
        token_node[token_nodes] = 0
        return mask_node, token_node

    def forward(self, x, edge_index):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(x, edge_index)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, x, edge_index):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
