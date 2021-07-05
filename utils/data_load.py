# -*- coding: utf-8 -*-
# @project：wholee_get_walks
# @author:caojinlei
# @file: data_lod.py
# @time: 2021/05/28

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


def get_base_seq(path):
    with open(path, 'r') as f:
        seq_list = []
        for lines in f.readlines():
            seq = lines.strip().split(';')[1].split(',')
            seq_list.append(seq)
    return seq_list




