import argparse
import copy
import csv

from data_utils.data_util import coarsen_graph
from data_utils.wrapper import Wrapper
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
import torch.multiprocessing
import warnings

from transform.posen import AddRandomWalkPE

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

criterion = nn.BCEWithLogitsLoss(reduction="none")


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
            cumulate_num_node = edge.max().item() + 1 if edge.numel() > 0 else 0
        coarse_edge.append(torch.cat(layer_edge, dim=1))
    return coarse_edge


def wrap_dataset(dataset, args, transform):
    wrapped_dataset = Wrapper(dataset)
    proj_matrices, super_adjs, node_dicts, pe_list = [], [], [], []
    for data in tqdm(dataset, desc="preprocess data"):
        proj_layer_matrices, super_layer_adjs, node_layer_dicts = (
            coarsen_graph(data, args.coarse_layer, args.coarse_rate))
        proj_matrices.append(proj_layer_matrices)
        super_adjs.append(super_layer_adjs)
        node_dicts.append(node_layer_dicts)
    wrapped_dataset.put_item((proj_matrices, super_adjs, node_dicts))
    # add random walk positional encoding
    pe_list = []
    coarse_adj = wrapped_dataset.super_adj
    for adj in tqdm(coarse_adj, desc="add PE"):
        adj = adj[1]
        num_nodes = adj.shape[0]
        pe = transform(adj, num_nodes)
        pe_list.append(pe)
    wrapped_dataset.put_pe(pe_list)
    return wrapped_dataset


def train(args, model, device, loader, optimizer):
    model.train()
    coarse_layer = args.coarse_layer
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        coarse_proj, coarse_batch = get_coarse_proj(batch, coarse_layer, device)
        coarse_edge = get_coarse_edge(batch, coarse_layer, device)
        pred = model(batch.x, batch.pe, coarse_edge, batch.edge_attr, coarse_batch, coarse_proj)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    coarse_layer = args.coarse_layer
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            coarse_proj, coarse_batch = get_coarse_proj(batch, coarse_layer, device)
            coarse_edge = get_coarse_edge(batch, coarse_layer, device)
            pred = model(batch.x, batch.pe, coarse_edge, batch.edge_attr, coarse_batch, coarse_proj)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def train_reg(args, model, device, loader, optimizer):
    model.train()
    coarse_layer = args.coarse_layer
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        coarse_proj, coarse_batch = get_coarse_proj(batch, coarse_layer, device)
        coarse_edge = get_coarse_edge(batch, coarse_layer, device)
        pred = model(batch.x, batch.pe, coarse_edge, batch.edge_attr, coarse_batch, coarse_proj)
        y = batch.y.view(pred.shape).to(torch.float64)
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred-y))/y.size(0)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    coarse_layer = args.coarse_layer
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        coarse_proj, coarse_batch = get_coarse_proj(batch, coarse_layer, device)
        coarse_edge = get_coarse_edge(batch, coarse_layer, device)
        with torch.no_grad():
            pred = model(batch.x, batch.pe, coarse_edge, batch.edge_attr, coarse_batch, coarse_proj)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse = np.sqrt(mse)
    if args.dataset in ['qm7', 'qm8', 'qm9']:
        result = mae
    if args.dataset in ['esol', 'freesolv', 'lipophilicity']:
        result = rmse
    return result


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument("--runseed", type=int, nargs="+", default=[0])
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument('--coarse_layer', type=int, default=2)
    parser.add_argument('--coarse_rate', type=float, default=0.5)
    args = parser.parse_args()

    args.use_early_stopping = args.dataset in ("muv")
    args.scheduler = args.dataset in ("bace")
    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'
    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    elif args.dataset == 'qm7':
        num_tasks = 1
    elif args.dataset == 'qm8':
        num_tasks = 12
    elif args.dataset == 'qm9':
        num_tasks = 12
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
                                                                    frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # print(train_dataset[0])
    transform = AddRandomWalkPE(10)
    train_dataset = wrap_dataset(train_dataset, args, transform)
    valid_dataset = wrap_dataset(valid_dataset, args, transform)
    test_dataset = wrap_dataset(test_dataset, args, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    result_path = f"results/{args.dataset}"
    result_filename = result_path + "/logging.csv"
    result_acc = result_path + "/acc.csv"
    os.makedirs(result_path, exist_ok=True)
    headerList = ["dataset", "coarse_rate", "acc", "std"]
    accHeader = ["dataset", "coarse_rate", "std_acc", "test_acc"]
    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)
    accs = []
    for seeds in args.runseed:

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []



        print(f"======run seed:{seeds}========")
        torch.manual_seed(seeds)
        np.random.seed(seeds)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds)

        # set up model
        model = GNN_graphpred(args.coarse_layer, args.num_layer, args.emb_dim, num_tasks, JK=args.JK,
                              drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)

        if not args.input_model_file == "":
            print("load pretrained model from:", args.input_model_file)
            model.from_pretrained(args.input_model_file)

        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = None

        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch))
            if task_type == 'cls':
                train(args, model, device, train_loader, optimizer)
            else:
                train_reg(args, model, device, train_loader, optimizer)
            if scheduler is not None:
                scheduler.step()

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            if task_type == 'cls':
                val_acc = eval(args, model, device, val_loader)
                test_acc = eval(args, model, device, test_loader)
            else:
                val_acc = eval_reg(args, model, device, val_loader)
                test_acc = eval_reg(args, model, device, test_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

        if args.use_early_stopping:
            final_val_acc = best_val_acc
            final_test_acc = best_test_acc
        else:
            final_val_acc = val_acc_list[-1]
            final_test_acc = test_acc_list[-1]

        with open(result_acc, "a+") as f:
            f.seek(0)
            header = f.read(6)
            if header != "dataset":
                dw = csv.DictWriter(f, delimiter=',',
                                    fieldnames=accHeader)
                dw.writeheader()
            acc_line = "{}, {}, {:.4f}, {:.4f}\n".format(args.dataset, args.coarse_rate, final_val_acc,
                                                         final_test_acc)
            f.write(acc_line)
        accs.append(final_test_acc)
    with open(result_filename, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "dataset":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()
        line = "{}, {}, {:.4f}, {:.4f}\n".format(args.dataset, args.coarse_rate, np.mean(accs),
                                                 np.std(accs))
        f.write(line)


if __name__ == "__main__":
    main()
