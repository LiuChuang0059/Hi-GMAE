from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)


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

    def forward(self, edge_index, num_nodes):
        row, col = edge_index
        num_edges = edge_index.size(1)
        N = num_nodes

        value = torch.ones(num_edges, device=row.device)

        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)

        return pe
