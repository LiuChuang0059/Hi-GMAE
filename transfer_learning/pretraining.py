import argparse
import os
from functools import partial

from transform.posen import AddRandomWalkPE
from data_utils.data_util import coarsen_graph
from data_utils.wrapper import Wrapper
from loader import MoleculeDataset
from dataloader import DataLoaderMasking, DataLoaderMaskingPred  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNNDecoder, MultiLayerEncoder, MultiLayerDecoder
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskAtom


from tensorboardX import SummaryWriter

import timeit


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


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


def train_mae(args, model_list, coarse_layer, loader, optimizer_list, device, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    model, dec_pred_atoms, dec_pred_bonds = model_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimizer_list

    model.train()
    dec_pred_atoms.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        en_feature_x = batch.x
        coarse_edge = get_coarse_edge(batch, coarse_layer, device)
        coarse_proj, coarse_batch = get_coarse_proj(batch, coarse_layer, device)
        for i in range(0, coarse_layer):
            if i == 0:
                proj = coarse_proj[i].to(device)
                en_feature_x = model.encoders[i](en_feature_x, coarse_edge[i], batch.edge_attr, True)  # encoder
                en_feature_x = torch.matmul(proj, en_feature_x)
            else:
                pe = torch.cat(batch.pe, dim=0)
                en_feature_x = model.encoders[i](en_feature_x, pe, coarse_batch[i], None, False)  # encoder

        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        de_feature_x = en_feature_x
        for i in range(coarse_layer - 1, -1, -1):
            if i == 0:
                de_feature_x = dec_pred_atoms.decoders[i](de_feature_x, coarse_edge[i], batch.edge_attr,
                                                          masked_node_indices)
            else:
                proj = coarse_proj[i - 1].to(device)
                de_feature_x = dec_pred_atoms.decoders[i](de_feature_x, coarse_edge[i], None, masked_node_indices)
                de_feature_x = torch.matmul(proj.T, de_feature_x)
        pred_node = de_feature_x
        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:, 0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        # acc_node_accum += acc_node

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = en_feature_x[masked_edge_index[0]] + en_feature_x[masked_edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep)
            loss += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])

            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum / step  # , acc_node_accum/step, acc_edge_accum/step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument('--coarse_layer', type=int, default=2)
    parser.add_argument('--coarse_rate', type=float, default=0.25)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    dataset_name = args.dataset
    # set up dataset and transform function.
    # dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform =
    # MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)
    # get 1000 sample
    wrapped_dataset = Wrapper(dataset)
    proj_matrices, super_adjs, node_dicts, pe_list = [], [], [], []
    path_coarse = f'./coarse/{args.coarse_rate}/'
    os.makedirs(path_coarse, exist_ok=True)
    path_proj = path_coarse + 'proj.pt'
    path_adjs = path_coarse + 'adjs.pt'
    path_dict = path_coarse + 'node_dict.pt'
    path_pe = path_coarse + 'pe.pt'
    if os.path.exists(path_proj) and os.path.exists(path_adjs) and os.path.exists(path_dict):
        proj_matrices = torch.load(path_proj)
        super_adjs = torch.load(path_adjs)
        node_dicts = torch.load(path_dict)
    else:
        for data in tqdm(dataset, desc="preprocess data"):
            proj_layer_matrices, super_layer_adjs, node_layer_dicts = (
                coarsen_graph(data, args.coarse_layer, args.coarse_rate))
            proj_matrices.append(proj_layer_matrices)
            super_adjs.append(super_layer_adjs)
            node_dicts.append(node_layer_dicts)
        torch.save(proj_matrices, path_proj)
        torch.save(super_adjs, path_adjs)
        torch.save(node_dicts, path_dict)
    wrapped_dataset.put_item((proj_matrices, super_adjs, node_dicts))
    transform = AddRandomWalkPE(10)
    if os.path.exists(path_pe):
        pe_list = torch.load(path_pe)
    else:
        coarse_adj = wrapped_dataset.super_adj
        for adj in tqdm(coarse_adj, desc="add PE"):
            adj = adj[1]
            num_nodes = adj.shape[0]
            pe = transform(adj, num_nodes)
            pe_list.append(pe)
        torch.save(pe_list, path_pe)
    wrapped_dataset.put_pe(pe_list)
    loader = DataLoaderMaskingPred(wrapped_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers,
                                   mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    model = MultiLayerEncoder(args.coarse_layer, args.num_layer, args.emb_dim, JK=args.JK,
                              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    # linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)
    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119  # + 3
    atom_pred_decoder = MultiLayerDecoder(args.coarse_layer, args.emb_dim, NUM_NODE_ATTR, JK=args.JK,
                                          gnn_type=args.gnn_type).to(device)
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model, atom_pred_decoder, bond_pred_decoder]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / args.epochs)) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, None]
    else:
        scheduler_model = None
        scheduler_dec = None

    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds]

    output_file_temp = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        # train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        # print(train_loss, train_acc_atom, train_acc_bond)

        train_loss = train_mae(args, model_list, args.coarse_layer, loader, optimizer_list, device,
                               alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 20 == 0:
                torch.save(model.state_dict(), output_file_temp + f"_{epoch}.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

    output_file = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")


if __name__ == "__main__":
    main()
