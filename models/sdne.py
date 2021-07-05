# -*- coding: utf-8 -*-
# @project：SDNE
# @author:caojinlei
# @file: sdne.py
# @time: 2021/06/21
import sys

import torch
from torch import nn
import numpy as np
import scipy.sparse as sp


class SDNE(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(SDNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        # Build Encoder
        modules = []
        for hidden in hidden_layers:
            modules.append(nn.Linear(input_dim,hidden))
            modules.append(nn.ReLU())
            input_dim = hidden
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        for hidden in reversed(hidden_layers[:-1]):
            modules.append(nn.Linear(input_dim,hidden))
            modules.append(nn.ReLU())
            input_dim = hidden
        modules.append(nn.Sequential(
            nn.Linear(input_dim, self.input_dim),
            nn.ReLU()
        ))
        self.decoder = nn.Sequential(*modules)

    def forward(self, A):
        """
        输入节点的邻接矩阵
        :param A:领接矩阵
        :return:
        """
        Z = self.encoder(A)
        A_hat = self.decoder(Z)
        return A_hat, Z


if __name__ == '__main__':
    import networkx as nx
    from utils.Logginger import init_logger
    from utils.losses import loss_reg, loss_function
    from data_load.data_load import ProductData
    import os
    from utils.utils import create_adjacency_laplace_matrix
    from torch.cuda.amp import autocast as at

    log = init_logger('SDNE', '../output/')
    # path = '../data/wiki/Wiki_edgelist.txt'
    path = '../data/wholee/pid_edges.csv'
    G = nx.read_edgelist(path,
                         create_using=nx.Graph(), nodetype=None, data=[('weight', 1.)])
    node_size = G.number_of_nodes()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SDNE(node_size, hidden_layers=[256, 128]).to(device)
    batch_size = 1024
    optimizer = torch.optim.Adam(model.parameters(),1e-4)
    dataset = ProductData(path, batch_size, 2)
    train_iter = dataset.data_load()
    alpha = 1e-5
    beta = 5
    v = 1e-5
    for epoch in range(5):
        model.train()
        loss_epoch = 0
        for index, batch in enumerate(train_iter):
            optimizer.zero_grad()
            batch_nodes_list = batch['nodes']
            A, L = create_adjacency_laplace_matrix(G, batch_nodes_list,device=device)
            # with at():
            A_hat, Z = model(A=A)
            print(sys.getsizeof(A_hat)/1024/1024)
            print(A_hat.dtype)
            loss_1st, loss_2nd = loss_function(A, A_hat, Z, alpha, beta, L)
            loss_reg_2 = loss_reg(model, v)
            loss = loss_1st + loss_2nd+loss_reg_2
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t'.format(
                epoch, index * len(batch_nodes_list), len(train_iter.dataset),
                       100. * index / len(train_iter), loss.item() / batch_size
            ))

