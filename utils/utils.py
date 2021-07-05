# -*- coding: utf-8 -*-
# @project：wholee_get_walks
# @author:caojinlei
# @file: utils.py
# @time: 2021/05/28
import torch
import numpy as np
from utils.Logginger import init_logger
import networkx as nx
from tqdm import tqdm
import random
from random import sample

logger = init_logger("utils", logging_path='output/')


def create_adjacency_laplace_matrix(G, batch_nodes_list, device, mode='Di', step='train'):
    """
    论文的核心,节点相连/不相连，s_ij=1/0，其实也可以按次数，两个节点高度相关，s_ij=边的次数
    :param G:
    :param batch_nodes_list:
    :param device:
    :param mode:
    :return:
    """
    node_size = G.number_of_nodes()
    nodes_all = list(G.nodes())
    length_node = len(batch_nodes_list)
    adjacency_matrix_data = torch.zeros([length_node, node_size])
    adjacency_matrix_data_ = np.zeros([length_node, length_node])
    # 领接矩阵[无向和有向一样]
    for node in batch_nodes_list:
        base_index = batch_nodes_list.index(node)
        if G.has_edge(node, node):
            adjacency_matrix_data[base_index][base_index] = 1.
        for i, _ in G[node].items():
            index = nodes_all.index(i)
            if base_index != index:
                adjacency_matrix_data[base_index][index] = 1.
    if step == 'predict':
        return adjacency_matrix_data.to(device)
    # 拉普拉斯矩阵:L=D-A
    else:
        if mode == 'Di':
            degree_list = []
            # 有向图的度矩阵为出度和入度之和
            for i in range(length_node):
                node = batch_nodes_list[i]
                degree = G.degree(node)
                if type(degree) != int:
                    degree = 0
                if G.has_edge(node, node):
                    degree = degree - 2.
                degree_list.append(degree)
                for j in range(length_node):
                    if i == j:
                        continue
                    else:
                        node2 = batch_nodes_list[j]
                        if G.has_edge(node, node2):
                            adjacency_matrix_data_[i][j] += 1.
                            adjacency_matrix_data_[j][i] += 1.
            deg_matrix = np.diag(degree_list)
        else:
            # 无向图的度矩阵为每行之和
            for i in range(length_node):
                node = batch_nodes_list[i]
                for j in range(length_node):
                    if i == j:
                        continue
                    else:
                        node2 = batch_nodes_list[j]
                        if G.has_edge(node, node2):
                            adjacency_matrix_data_[i][j] = 1.
            deg_matrix = np.diag(np.sum(adjacency_matrix_data_, axis=0))
        laplace_matrix = torch.from_numpy(deg_matrix - adjacency_matrix_data_).float()
        return adjacency_matrix_data.to(device), laplace_matrix.to(device)


def edges_sample(G, sample_frac):
    edges_list = list(G.edges())
    pop = int(len(edges_list) * sample_frac)
    test_pos_edges = random.sample(edges_list, pop)
    return test_pos_edges


def get_negative_samples(G, test_pos_list):
    count = 0
    nodes_list = list(G.nodes())
    random.shuffle(nodes_list)
    break_number = len(test_pos_list)
    test_neg_list = []
    for x in range(100):
        for i in range(len(nodes_list)):
            cur_node = nodes_list[i]
            random_node = random.choice(nodes_list)
            while random_node in list(G.adj[cur_node]):
                logger.info(f'{cur_node}-{random_node} is positive sample')
                random_node = random.choice(nodes_list)
            test_neg_list.append((cur_node, random_node))
            count += 1
            if count % 100000 == 0:
                logger.info(f'negative sample {count}')
            if count >= break_number:
                break
        if count >= break_number:
            break
    return test_neg_list


def get_lp_data(pos_list, neg_list):
    l = pos_list + neg_list
    return list(set(np.array(l).flatten()))
