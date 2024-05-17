import csv
import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def get_mask_edge(edge_index, mask_nodes):
    mask_edge_index = edge_index.clone()
    mask_node = torch.where(mask_nodes == 0)[0]
    mask = torch.isin(edge_index, mask_node)
    mask_edge_index = mask_edge_index[:, ~mask.any(dim=0)]

    return mask_edge_index


def get_coarse_proj(batch_g, coarse_layer, device):
    coarse_proj = []
    coarse_batch = [batch_g.batch]
    num_graph = len(batch_g.proj)
    # get mask projection
    for i in range(1, coarse_layer):
        layer_proj = []
        layer_batch = []
        cnt = 0
        for j in range(num_graph):
            layer_proj.append(batch_g.proj[j][i])
        mat_size = [proj.size() for proj in layer_proj]
        rows = sum([size[0] for size in mat_size])
        cols = sum([size[1] for size in mat_size])
        concat_mat = torch.zeros((rows, cols)).to(device)
        row_start, col_start = 0, 0
        for g_mat in layer_proj:
            row_end = row_start + g_mat.size()[0]
            col_end = col_start + g_mat.size()[1]
            concat_mat[row_start:row_end, col_start:col_end] = g_mat
            row_start = row_end
            col_start = col_end
        coarse_proj.append(concat_mat)
        for size in mat_size:
            row = size[0]
            layer_batch.append(torch.zeros(row, dtype=torch.long, device=device) + cnt)
            cnt += 1
        coarse_batch.append(torch.cat(layer_batch, dim=0))
    return coarse_proj, coarse_batch


def get_coarse_edge(batch_g, coarse_layer, device):
    # get edge info
    coarse_edge = [batch_g.edge_index]
    num_graph = len(batch_g.proj)
    for i in range(1, coarse_layer):
        layer_edge = []
        cumulate_num_node = 0
        for j in range(num_graph):
            adj = batch_g.super_adj[j][i]
            edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long, device=device)
            layer_edge.append(edge_index)
        for edge in layer_edge:
            edge[0] += cumulate_num_node
            edge[1] += cumulate_num_node
            cumulate_num_node = edge.max().item() + 1  # get max node index
        coarse_edge.append(torch.cat(layer_edge, dim=1))
    return coarse_edge


def get_mask_list(mask_nodes, token_nodes, coarse_proj, coarse_layer, device):
    mask_nodes_list = []
    token_nodes_list = []
    mask_nodes_list.append(mask_nodes)
    token_nodes_list.append(token_nodes)
    for i in range(coarse_layer - 1, 0, -1):
        proj = coarse_proj[i - 1].to(device)
        next_mask_node = torch.matmul(proj.T, mask_nodes_list[-1])
        next_token_node = torch.matmul(proj.T, token_nodes_list[-1])
        mask_nodes_list.append(torch.tensor(next_mask_node, device=device))
        token_nodes_list.append(torch.tensor(next_token_node, device=device))
    mask_nodes_list.reverse()
    token_nodes_list.reverse()
    return mask_nodes_list, token_nodes_list


def get_encoder_out(batch, encoders, x, pooler, coarse_edge, coarse_proj, coarse_batch, coarse_layer, last_enc, device):
    res = []
    en_feature_x = x.clone()
    for i in range(1, coarse_layer + 1):
        edge_index = coarse_edge[i - 1]
        if i != coarse_layer or last_enc != "transformer":
            en_feature_x, _ = encoders[i - 1](en_feature_x, edge_index, return_hidden=True)
        else:
            en_feature_x = encoders[i - 1](en_feature_x, batch.pe[0], coarse_batch[i - 1])
        res.append(en_feature_x)
        if i != coarse_layer:
            proj = coarse_proj[i - 1].to(device)
            en_feature_x = torch.matmul(proj, en_feature_x)
    for i in range(len(res)):
        if pooler == "max":
            res[i] = global_max_pool(res[i], coarse_batch[i])
        elif pooler == "mean":
            res[i] = global_mean_pool(res[i], coarse_batch[i])
        elif pooler == "sum":
            res[i] = global_add_pool(res[i], coarse_batch[i])
        elif pooler == 'mean_max':
            mean_pool = global_mean_pool(res[i], coarse_batch[i])
            max_pool = global_max_pool(res[i], coarse_batch[i])
            res[i] = torch.cat([mean_pool, max_pool], dim=1)
        elif pooler == 'sum_max':
            sum_pool = global_add_pool(res[i], coarse_batch[i])
            max_pool = global_max_pool(res[i], coarse_batch[i])
            res[i] = torch.cat([sum_pool, max_pool], dim=1)
        else:
            raise NotImplementedError
    out = torch.sum(torch.stack(res), dim=0)  # readout: add
    return out


def get_layer_feature(x, coarse_proj, coarse_layer, device):
    en_feature_x = x.clone()
    layer_feature = [en_feature_x]
    for i in range(1, coarse_layer):
        proj = coarse_proj[i - 1].to(device)
        en_feature_x = torch.matmul(proj, en_feature_x)
        layer_feature.append(en_feature_x)
    return layer_feature


def get_layer_loss(model, enc_x, dec_x, mask_nodes, final_layer):
    if final_layer is False:
        dec_x = model.proj(dec_x)
    mask = torch.where(mask_nodes == 0)[0]
    loss = model.criterion(enc_x[mask], dec_x[mask])
    return loss


def recover_mask(mask_node_list, token_node_list, coarse_layer, recover_rate=0.1):
    # recover mask node
    for i in range(0, coarse_layer - 1):
        zero_index = torch.where(mask_node_list[i] == 0)[0]
        recover_num = int(len(zero_index) * recover_rate)
        recover_index = torch.randperm(len(zero_index))[:recover_num]
        mask_node_list[i][zero_index[recover_index]] = 1
        token_node_list[i][zero_index[recover_index]] = 1
    return mask_node_list, token_node_list


def adjust_recover_rate(recover_rate, epoch, max_epoch, gamma):
    if recover_rate > 0:
        decay = (1 - epoch / max_epoch)
        decay = pow(decay, gamma)
        recover_rate = recover_rate * decay
    return recover_rate


def save_result(args, acc, std):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    headerList = ["dataset", "num_layers", "mask_rate", "gamma", "end_epoch", "mask_edge", "recover_rate",
                  "coarse_layer", "coarse_rate", "method", "result"]
    with open(f"./results/{args.dataset}.csv", "a+") as f:
        f.seek(0)
        first_line = f.readline()
        if first_line == "":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()
        line = "{},{},{},{},{},{},{},{},{}, {}, {:.4f}±{:.4f}".format(
            args.dataset, args.num_layers, args.mask_rate, args.gamma, args.epoch_rate * args.max_epoch,
            args.mask_edge, args.recover_rate, args.coarse_layer, args.coarse_rate, args.coarse_type, acc, std)

        f.write(line + "\n")


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.05)

    parser.add_argument("--encoder", type=str, default="gin")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    # for graph coarsening
    parser.add_argument("--coarse_layer", type=int, default=2)
    parser.add_argument("--coarse_rate", type=float, default=0.5)
    parser.add_argument('--mask_edge', action="store_true", default=False)
    parser.add_argument("--coarse_type", type=str, default="algebraic_JC")
    parser.add_argument("--recover_rate", type=float, default=-1)
    parser.add_argument("--last_enc", type=str, default="transformer")
    parser.add_argument("--pe_dim", type=int, default=20)
    # for recovery
    parser.add_argument("--epoch_rate", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
