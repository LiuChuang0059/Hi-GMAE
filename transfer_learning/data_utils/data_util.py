from collections import Counter
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from pygsp import graphs
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree

from ogb.nodeproppred import PygNodePropPredDataset

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_utils.graph_coarsening.coarsen_utils import coarsen
from data_utils.graph_coarsening.graph_utils import *
from data_utils.wrapper import Wrapper
import warnings

from transform.posen import AddRandomWalkPE

warnings.filterwarnings("ignore")


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


def process_data(num_node, adj, rate):  # adj is a sparse matrix
    g = graphs.Graph(adj)

    c, _, _, _ = coarsen(g, K=10, r=rate, method='greedy')
    c = c / c.sum(1)
    c = torch.tensor(c.toarray(), dtype=torch.float32)
    node_dict = {}
    for i in range(num_node):
        node_dict[i] = torch.where(c[:, i] > 0)[0].item()
    node_dict = reverse_dict(node_dict)  # {super_node: [node1, node2, ...]}
    coarse_adj = coarse_graph_adj(adj, c)  # [num_super_node, num_super_node]
    return c, coarse_adj, node_dict


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def coarsen_graph(data, coarse_layer, rate):
    node_layer_dicts = []
    super_layer_adjs = []
    proj_layer_matrices = []
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                        shape=(data.num_nodes, data.num_nodes),
                        dtype=np.float32)
    # initialize the coarse input
    proj_layer_matrices.append([])
    node_layer_dicts.append([])
    super_layer_adjs.append(adj)
    for i in range(1, coarse_layer):
        last_layer_adj = zero_diag(super_layer_adjs[-1])
        proj_matrix, super_adj, node_dict = process_data(last_layer_adj.shape[0], last_layer_adj, rate)
        # to_undirected
        super_adj = np.maximum(super_adj, super_adj.T)
        # transform the super_adj to coo format
        super_adj = sp.coo_matrix(super_adj)
        proj_layer_matrices.append(proj_matrix)
        super_layer_adjs.append(super_adj)
        node_layer_dicts.append(node_dict)
    """
    for i in range(coarse_layer):
        G = nx.to_networkx_graph(super_layer_adjs[i])
        pos_G = nx.spring_layout(G) 
        nx.draw(G, pos_G, with_labels=True, node_size=300, node_color='skyblue', font_size=10)
        plt.title("Graph G")
        plt.show()
    """
    return proj_layer_matrices, super_layer_adjs, node_layer_dicts
