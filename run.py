# -*- coding: utf-8 -*-
# @project：SDNE
# @author:caojinlei
# @file: run.py
# @time: 2021/06/22
import argparse
import networkx as nx
import os
from utils.classify import NodeClassify, LinkPredict, MultiClassifier
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.model_selection import train_test_split
from models.sdne import SDNE
from utils.Logginger import init_logger
from utils.lambda_lr import warmup_lr_scheduler
from utils.losses import loss_reg, loss_function
from data_load.data_load import *
from callback.modelcheckpoint import ModelCheckPoint
from torch.cuda.amp import autocast as at
from utils.utils import *


def arguments():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('-a', '--alpha', type=float, default=1e-5, metavar='N',
                        help='Parameters that control the 1st order loss(default: 1e-5)')
    parser.add_argument('-b', '--beta', type=int, default=5, metavar='N',
                        help='The parameters controlling the second-order loss have higher penalty '
                             'coefficients for non-zero elements(default: 5)')
    parser.add_argument('--v', type=float, default=1e-5, metavar='N',
                        help='Controls the parameters of the regularization term(default: 1e-5)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='Optimizer parameter(default: 1e-3)')
    parser.add_argument('--method', type=str, default='n',
                        help='Classify_method : node classify(n)/link(lp).(default: n)')
    parser.add_argument('-sf', '--sample_frac', type=float, default=0.2, metavar='N',
                        help='Test size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training(default: False)')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mode', type=str, default='Di', metavar='S',
                        help='The type of graph(default: Di)')
    parser.add_argument('--save', type=str, default='y', metavar='S',
                        help='Whether or not to save(default: y)')
    parser.add_argument('--train', type=str, default='y', metavar='S',
                        help='Train or predict(default: y)')
    parser.add_argument('--module', type=str, default='SDNE', metavar='N',
                        help='Which model to choose(default: SDNE)')
    parser.add_argument('--log-interval', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status(default: 1024)')
    return parser.parse_args()


def train(model, batch_size, alpha, beta, v, epoch, train_iter, log, optimizer, scheduler, G, device, mode):
    model.train()
    train_loss = 0
    for index, batch in enumerate(train_iter):
        optimizer.zero_grad()
        batch_nodes_list = batch['nodes']
        A, L = create_adjacency_laplace_matrix(G, batch_nodes_list, device, mode)
        A_hat, Z = model(A=A)
        loss_1st, loss_2nd = loss_function(A, A_hat, Z, alpha, beta, L)
        loss_reg_2 = loss_reg(model, v)
        loss = loss_1st + loss_2nd + loss_reg_2
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
        log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t'.format(
            epoch, index * batch_size + len(batch_nodes_list), len(train_iter.dataset),
                   100 * (index * batch_size + len(batch_nodes_list)) / len(train_iter.dataset),
                   loss.item() / len(batch_nodes_list)
        ))
    log.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_iter.dataset)))
    result_loss = train_loss / len(train_iter.dataset)
    return result_loss


@torch.no_grad()
def get_embedding(model, train_iter, G, device, mode):
    model.eval().to(device)
    embedding_dict = {}
    for index, batch in enumerate(train_iter):
        batch_nodes_list = batch['nodes']
        A = create_adjacency_laplace_matrix(G, batch_nodes_list, device, mode, step='predict')
        embed = model.encoder(A)
        for i, embedding in enumerate(embed.cpu().numpy()):
            embedding_dict[batch_nodes_list[i]] = embedding
        log.info('Get embedding: [{}/{} ({:.0f}%)]\t'.format(index * len(batch_nodes_list), len(train_iter.dataset),
                                                             100 * index * len(batch_nodes_list) / len(
                                                                 train_iter.dataset)))
    return embedding_dict


def evaluate(X, Y, embedding_dict, test_size):
    base_model = LogisticRegression(solver='sag', n_jobs=3, max_iter=100000)
    node_classify = NodeClassify(X, Y, embedding_dict, base_model, test_size)
    node_classify.train()
    score = node_classify.evaluate()
    return score


def cmd_entry(args, log):
    # 获取参数
    path = 'data/wiki/Wiki_edgelist.txt'
    nodes_path = 'data/wiki/wiki_labels.txt'
    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta
    v = args.v
    lr = args.lr
    mode = args.mode
    classify_method = args.method
    # 用点击序构造底层图
    if mode == 'Di':
        G = nx.read_edgelist(path,
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    else:
        G = nx.read_edgelist(path,
                             create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
    click_pro = len(G.nodes())
    if classify_method == 'lp':
        # 构造测试样本[链路预测]
        test_pos_list = edges_sample(G, sample_frac=0.15)
        test_neg_list = get_negative_samples(G, test_pos_list)
        X_lp = get_lp_data(test_pos_list, test_neg_list)
        lp_dataset = ProductData(X_lp, batch_size, 2)
        lp_iter = lp_dataset.data_load()
        G.remove_edges_from(test_pos_list)
    node_size = G.number_of_nodes()

    # 构造训练集合[节点分类]
    X, Y = get_nodes_class(nodes_path)
    # length_x = sample(range(len(X1)),2000)
    # X = [X1[x] for x in length_x]
    # Y = [Y1[x] for x in length_x]
    train_dataset = ProductData(X, batch_size, 2)
    train_iter = train_dataset.data_load()

    # 训练模型
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:1" if args.cuda else "cpu")
    model = SDNE(node_size, hidden_layers=[256, 128]).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_factor = 1. / 1000
    warmup_iters = min(1000, len(train_iter) - 1)
    scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 是否保存模型
    if args.save == 'y':
        save_module = ModelCheckPoint(model=model, optimizer=optimizer, args=args, log=log)
        state = save_module.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, batch_size, alpha, beta, v, epoch, train_iter, log, optimizer, scheduler, G, device,
                           mode)
        if args.save == 'y':
            state = save_module.save_step(state, train_loss)

    # 评估模型[lp/node]
    if classify_method == 'lp':
        embedding_dict_lp = get_embedding(model, lp_iter, G, device, mode)
        link_predict = LinkPredict(test_pos_list, test_neg_list, embedding_dict_lp)
        pos_sim_list, neg_sim_list = link_predict.train()
        pos_sim_list = sorted(pos_sim_list, reverse=True)
        neg_sim_list = sorted(neg_sim_list, reverse=True)
        topk_list = [50, 100, 150, 200, 250]
        score = link_predict.evaluate(pos_sim_list, neg_sim_list, topk_list)
        log.info(f'score_lp:{score}')
    else:
        embedding_dict = get_embedding(model, train_iter, G, device, mode)
        score = evaluate(X, Y, embedding_dict, test_size=args.sample_frac)
        log.info(f'score_n:{score}')
    return score


if __name__ == '__main__':
    log = init_logger('SDNE', 'output/')
    args = arguments()
    print(args)
    score = cmd_entry(args, log)
