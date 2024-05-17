import copy

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from graphmae.utils import create_optimizer, accuracy


def get_encoder_out(encoders, x, graph, coarse_layer, device):
    res = []
    for i in range(1, coarse_layer + 1):
        adj = graph.super_adj[i - 1]
        edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long, device=device)
        x, _ = encoders[i - 1](x, edge_index, return_hidden=True)
        res.append(x)
        if i != coarse_layer:
            proj = graph.proj[i].to(device)
            x = torch.matmul(proj, x)
    for i in range(1, coarse_layer):
        for j in range(i, 0, -1):
            proj = graph.proj[j].to(device)
            res[i] = torch.matmul(proj.T, res[i])
    out = torch.cat(res, dim=-1)
    return out


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                   coarse_layer, mute=False):
    model.eval()
    """
    if linear_prob:
        with torch.no_grad():
            x = model.embed(x.to(device), graph.edge_index.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)
    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for fine-tuning: {sum(num_finetune_params)}")
    """
    with torch.no_grad():
        x = get_encoder_out(model.encoders, x, graph, coarse_layer, device)
        in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f,
                                                                              max_epoch_f,  device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    x = feat.to(device)

    train_mask = graph.dataset.train_mask
    val_mask = graph.dataset.val_mask
    test_mask = graph.dataset.test_mask
    labels = graph.dataset.y.to(device)

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x, *args):
        logits = self.linear(x)
        return logits
