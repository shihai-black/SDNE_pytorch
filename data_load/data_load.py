# -*- coding: utf-8 -*-
# @project：wholee_get_walks
# @author:caojinlei
# @file: data_lod.py
# @time: 2021/05/28
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import torch.distributed as dist
import random


class MyDataSet(Dataset):
    def __init__(self, nodes):
        self.nodes = nodes
        # G = nx.read_edgelist(path,
        #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
        nodes_list = self.nodes
        self.data = [{'nodes': nodes_list[i], 'indexs': i} for i in range(len(nodes_list))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ProductData:
    def __init__(self, nodes, batch_size, num_works, shuffle=True, pin_memory=True):
        self.nodes = nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_works = num_works
        self.pin_memory = pin_memory

    def data_load(self):
        dataset = MyDataSet(self.nodes)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2,
                                 pin_memory=self.pin_memory)
        return data_loader

    # def data_load_dist(self):
    #     dist.init_process_group(backend='nccl',init_method='env://')
    #     dataset = MyDataSet(self.path)
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     data_laod = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2,
    #                              pin_memory=self.pin_memory,sampler=sampler)
    #     return data_laod


def get_nodes_class(path):
    with open(path, 'r') as f:
        nodes_list = []
        classes_list = []
        for lines in f.readlines():
            node = lines.strip().split(' ')[0]
            classes = lines.strip().split(' ')[1]
            nodes_list.append(node)
            classes_list.append(classes)
    return nodes_list, classes_list


if __name__ == '__main__':
    nodes_path = '../data/wiki/wiki_labels.txt'
    X, Y = get_nodes_class(nodes_path)
    dataset = ProductData(X, 64, 2)
    data_load = dataset.data_load()
    for index, batch in enumerate(data_load):
        nodes_list = batch['nodes']
        print(nodes_list)
