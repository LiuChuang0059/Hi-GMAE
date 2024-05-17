import numpy as np
import torch

from graph_coarsening.coarsen_utils import coarsen
from pygsp import graphs


def coarse_graph_adj(mx, p):
    p[p > 0] = 1.
    p = np.array(p)
    rowsum = p.sum(1)
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    p = np.matmul(p.T, r_mat_inv)
    mx = np.matmul(mx.toarray(), p)
    mx = np.matmul(p.T, mx)
    return mx


def reverse_dict(node_dict):
    new_dict = {}
    for key, value in node_dict.items():
        if value in new_dict:
            new_dict[value].append(key)
        else:
            new_dict[value] = [key]
    return new_dict


def process_data(num_node, node_feature, adj, rate, method):  # adj is a sparse matrix
    g = graphs.Graph(adj)

    c, _, _, _ = coarsen(g, K=10, r=rate, method=method)
    c = c / c.sum(1)
    c = torch.tensor(c.toarray(), dtype=torch.float32)
    super_node_feature = torch.matmul(c, node_feature.float())  # [num_super_node, num_feature]
    node_dict = {}
    for i in range(num_node):
        node_dict[i] = torch.where(c[:, i] > 0)[0].item()
    node_dict = reverse_dict(node_dict)  # {super_node: [node1, node2, ...]}
    coarse_adj = coarse_graph_adj(adj, c)  # [num_super_node, num_super_node]
    return c, super_node_feature, coarse_adj, node_dict
