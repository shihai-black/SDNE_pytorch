# -*- coding: utf-8 -*-
# @project：SDNE
# @author:caojinlei
# @file: lambda_lr.py
# @time: 2021/06/24
import torch


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    热启动三角函数，优化器学习旅先增后降，快速收敛
    :param optimizer:
    :param warmup_iters:
    :param warmup_factor:
    :return:
    """
    def f(x):  # x是step次数
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters  # 当前进度 0-1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
