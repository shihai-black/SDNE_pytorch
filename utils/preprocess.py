# -*- coding: utf-8 -*-
# @project：wholee_get_walks
# @author:caojinlei
# @file: preprocess.py
# @time: 2021/05/27
from Logginger import init_logger
from tqdm import tqdm
from random import sample

logger = init_logger("get edges and nodes", logging_path='../output/')


def get_data(path, edges_path, nodes_path):
    with open(path, 'r') as f:
        edges_list = []
        nodes_list = []
        for line in f.readlines():
            length = int(line.split(';')[0])
            pid_walks_list = line.strip().split(';')[1].split(',')
            nodes_class_list = eval(line.strip().split(';')[2])
            nodes_list.append(pid_walks_list[length - 1] + ' ' + nodes_class_list[length - 1])
            for i in range(length - 1):
                nodes_class = pid_walks_list[i] + ' ' + nodes_class_list[i]
                nodes_list.append(nodes_class)
                if pid_walks_list[i] == pid_walks_list[i + 1]:
                    continue
                else:
                    edge = pid_walks_list[i] + ' ' + pid_walks_list[i + 1]
                    edges_list.append(edge)
        edges_set = set(edges_list)
        nodes_set = set(nodes_list)
    logger.info(f'The process completes and fetch nodes :{len(nodes_set)}/ edges:{len(edges_set)}')
    with open(edges_path, 'w') as f:
        for edge in edges_set:
            f.write(edge)
            f.write('\n')
    logger.info('write edges_set')
    with open(nodes_path, 'w') as f:
        for node in nodes_set:
            f.write(node)
            f.write('\n')
    logger.info('write nodes_set')


def get_big_edges(input_path, out_path):
    with open(input_path, 'r') as f:
        edges_list = []
        for line in tqdm(f.readlines()):
            pid_walks_list = line.strip().split(',')
            length = len(pid_walks_list)
            for i in range(length - 1):
                edge = pid_walks_list[i] + ' ' + pid_walks_list[i + 1]
                edges_list.append(edge)
    logger.info(f'The process completes and fetch edges:{len(edges_list)}')
    with open(out_path, 'w') as f:
        for edge in edges_list:
            f.write(edge)
            f.write('\n')
    logger.info('write edges list')


def get_big_attrs(input_path, out_path):
    with open(input_path, 'r') as f:
        nodes_list = []
        for line in tqdm(f.readlines()):
            pid = line.strip().split('|')[0].split(':')[-1]
            cate1 = line.strip().split('|')[1].split(':')[-1]
            node = pid + ' ' + cate1
            nodes_list.append(node)
    logger.info(f'The process completes and fetch nodes:{len(nodes_list)}')
    with open(out_path, 'w') as f:
        for node in nodes_list:
            f.write(node)
            f.write('\n')
    logger.info('write node list')


def get_click_big_nodes(edges_path, nodes_path, out_path):
    with open(nodes_path, 'r') as f:
        nodes_dict = {}
        for line in tqdm(f.readlines()):
            pid = line.strip().split('|')[0].split(':')[-1]
            cate1 = line.strip().split('|')[1].split(':')[-1]
            nodes_dict[pid] = cate1
    with open(edges_path, 'r') as f:
        nodes2cate = []
        for line in tqdm(f.readlines()):
            pid_walks_list = line.strip().split(',')
            length = len(pid_walks_list)
            for i in range(length):
                pid = pid_walks_list[i]
                try:
                    cate1 = nodes_dict[pid]
                except Exception as e:
                    cate1 = 'nocate1'
                node = pid + ' ' + cate1
                nodes2cate.append(node)
    re_list = list(set(nodes2cate))
    with open(out_path, 'w') as f:
        for node in re_list:
            f.write(node)
            f.write('\n')
    logger.info('write node list')


def balance_nodes(click_nodes,click_sample_nodes_path,sample_size=500):
    """取出节点均衡的数据"""
    # 获取每个节点的类目信息
    classes_nodes = {}
    with open(click_nodes, 'r') as f:
        count = 0
        for lines in f.readlines():
            count += 1
            node = lines.strip().split(' ')[0]
            classes = lines.strip().split(' ')[1]
            if classes == 'nocate1':
                continue
            else:
                if classes_nodes.get(classes):
                    classes_nodes[classes].append(node)
                else:
                    classes_nodes[classes] = [node]
    # 获取每个类目的节点数量
    classes_length = {}
    for classes, nodes_list in classes_nodes.items():
        length_class = len(nodes_list)
        if (length_class >= sample_size) & (length_class / count >= 0.002):  # 取出节点数量>=500,且数据占比>2e-3
            classes_length[classes] = [length_class]
    # 从每个类目中抽取500个节点信息
    re_list = []
    for classes,_ in tqdm(classes_length.items()):
        nodes_list = classes_nodes[classes]
        nodes_sample = sample(nodes_list,sample_size)
        for node in nodes_sample:
            re = node + ' ' + classes
            re_list.append(re)
    with open(click_sample_nodes_path, 'w') as f:
        for re in re_list:
            f.write(re)
            f.write('\n')
    logger.info('write sample node list')

    return re_list


if __name__ == '__main__':
    mode = 3
    if mode == 1:
        path = '../input/pid_walks.csv'
        edges_path = '../input/pid_edges.csv'
        nodes_path = '../input/pid_nodes.csv'
        get_data(path, edges_path, nodes_path)
    elif mode == 2:
        input_path1 = '../data/wholee/pid_seq_day620.txt'
        output_path1 = '../data/wholee/pid_edges_big.txt'
        get_big_edges(input_path1, output_path1)
        input_path2 = '../data/wholee/pid_info.txt'
        output_path2 = '../data/wholee/pid_nodes_big.txt'
        get_big_attrs(input_path2, output_path2)
    else:
        edges = '../data/wholee/pid_seq_day620.txt'
        nodes = '../data/wholee/pid_info.txt'
        output_path = '../data/wholee/pid_click_nodes.txt'
        get_click_big_nodes(edges, nodes, output_path)
