from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)


def get_self_loop_attr(edge_index: Tensor, edge_attr: OptTensor = None,
                       num_nodes: Optional[int] = None) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes,) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


class AddRandomWalkPE:
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(
            self,
            walk_length: int) -> None:
        self.walk_length = walk_length

    def __call__(self, edge_index: Tensor, num_nodes: Optional[int] = None,
                 num_edges: Optional[int] = None) -> Tensor:
        return self.forward(edge_index, num_nodes)

    def forward(self, adj, num_nodes):

        adj = SparseTensor(row=torch.LongTensor(adj.row),
                           col=torch.LongTensor(adj.col),
                           value=torch.FloatTensor(adj.data),
                           sparse_sizes=(adj.shape[0], adj.shape[1]))
        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
        pe = torch.stack(pe_list, dim=-1)

        return pe
