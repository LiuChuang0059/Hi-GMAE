import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr is None:
            return self.propagate(edge_index, x=x, edge_attr=None)
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GCNConv, self).__init__(**kwargs)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        # x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

    # added
    def update(self, aggr_out):
        return self.linear(aggr_out)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, pe_dim, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(Transformer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.node_add = nn.Linear(in_dim, hidden_size - pe_dim)

    def forward(self, x, pe, coarse_batch, mask=None, attn_bias=None):
        if mask is None:
            mask = torch.ones(x.size(0), dtype=torch.long, device=x.device)
        keep_nodes = torch.where(mask == 1)[0]
        mask_nodes = torch.where(mask == 0)[0]
        final_x = torch.zeros_like(x, device=x.device)
        keep_x = x[keep_nodes]
        keep_pe = pe[keep_nodes]
        keep_batch = coarse_batch[keep_nodes]
        mask_x = x[mask_nodes]
        keep_x = self.node_add(keep_x)
        keep_x = torch.cat([keep_x, keep_pe], dim=-1)
        keep_x, mask = to_dense_batch(keep_x, keep_batch)
        y = self.self_attention_norm(keep_x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        keep_x = keep_x + y
        keep_x = keep_x[mask]
        y = self.ffn_norm(keep_x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        keep_x = keep_x + y
        final_x[keep_nodes] = keep_x
        final_x[mask_nodes] = mask_x
        return final_x


class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            elif gnn_type == "transformer":
                self.gnns.append(Transformer(emb_dim, 10, emb_dim, emb_dim,
                                             drop_ratio, drop_ratio, 4))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, embed = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 2:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            embed = argv[1]
        elif len(argv) == 5:
            x, pe, batch, mask, embed = argv[0], argv[1], argv[2], argv[3], argv[4]
        else:
            raise ValueError("unmatched number of arguments.")
        if embed:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        if len(argv) == 4:
            for layer in range(self.num_layer):
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_list.append(h)
        else:
            for layer in range(self.num_layer):
                h = self.gnns[layer](h_list[layer], pe, batch, mask, None)
                h = self.batch_norms[layer](h)
                # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_list.append(h)
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class MultiLayerEncoder(torch.nn.Module):
    def __init__(self, coarse_layer, num_layer, emb_dim, JK="last", drop_ratio=0.,
                 gnn_type="gin", last_type="transformer"):
        super().__init__()
        self._coarse_layer = coarse_layer
        self._num_layer = num_layer
        self._emb_dim = emb_dim
        self._JK = JK
        self._drop_ratio = drop_ratio
        self._gnn_type = gnn_type
        self.encoders = torch.nn.ModuleList()
        for i in range(coarse_layer):
            if i == 0:
                encoder = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
            else:
                encoder = GNN(1, emb_dim, JK, drop_ratio, gnn_type=last_type)
            self.encoders.append(encoder)


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr="add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr="add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2

    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            # x = self.enc_to_dec(x)
            # x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out


class MultiLayerDecoder(torch.nn.Module):
    def __init__(self, coarse_layer, hidden_dim, out_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self._coarse_layer = coarse_layer
        self._hidden_dim = hidden_dim
        self._JK = JK
        self._drop_ratio = drop_ratio
        self._gnn_type = gnn_type
        self.decoders = torch.nn.ModuleList()
        for i in range(coarse_layer):
            if i == 0:
                decoder = GNNDecoder(hidden_dim, out_dim, JK, drop_ratio, gnn_type=gnn_type)
            else:
                decoder = GNNDecoder(hidden_dim, hidden_dim, JK, drop_ratio, gnn_type=gnn_type)
            self.decoders.append(decoder)


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, coarse_layer, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.coarse_layer = coarse_layer

        self.gnn = MultiLayerEncoder(coarse_layer, num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type,
                                     last_type="transformer")

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location='cpu'))

    def forward(self, *argv):
        if len(argv) == 6:
            x, pe_list, coarse_edge, edge_attr, coarse_batch, coarse_proj = (argv[0], argv[1], argv[2], argv[3],
                                                                             argv[4], argv[5])
        else:
            raise ValueError("unmatched number of arguments.")

        """
        node_representation = self.gnn(x, edge_index, edge_attr)
        return self.graph_pred_linear(self.pool(node_representation, batch))
        """
        res = []
        super_feat = [x]
        for i in range(self.coarse_layer):
            en_feature_x = super_feat[-1]
            if i == 0:
                en_feature_x = self.gnn.encoders[i](en_feature_x, coarse_edge[i], edge_attr, True)
                res.append(en_feature_x)
                en_feature_x = torch.matmul(coarse_proj[i], en_feature_x)
            else:
                en_feature_x = self.gnn.encoders[i](en_feature_x, pe_list, coarse_batch[i], None, False)
                res.append(en_feature_x)
            super_feat.append(en_feature_x)

        # pooling
        for i in range(self.coarse_layer):
            res[i] = self.pool(res[i], coarse_batch[i])
        # concat
        graph_representation = torch.sum(torch.stack(res), dim=0)
        return self.graph_pred_linear(graph_representation)


if __name__ == "__main__":
    pass
