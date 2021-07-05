# -*- coding: utf-8 -*-
# @project：SDNE
# @author:caojinlei
# @file: losses.py
# @time: 2021/06/21
import torch


def loss_function(A, A_hat, Z, alpha, beta, L):
    """
    1阶+2阶损失函数
    :param A:邻接矩阵
    :param A_hat:输出的邻接矩阵
    :param Z:中间输出
    :return:
    """
    # 2阶损失
    beta_matrix = torch.ones_like(A)
    mask = A != 0
    beta_matrix[mask] = beta  # 主要目的我理解是让A_hat中非0元素保持非0
    loss_2nd = torch.mean(torch.sum(torch.pow((A - A_hat) * beta_matrix, 2), dim=1))
    # 1阶损失
    loss_1st = alpha * 2 * torch.trace(torch.matmul(torch.matmul(Z.transpose(0, 1), L), Z))
    return loss_1st, loss_2nd


def loss_reg(model, v):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = torch.mean(torch.sum(torch.square(param)))
            reg_loss += l2_reg
    return v / 2 * reg_loss

