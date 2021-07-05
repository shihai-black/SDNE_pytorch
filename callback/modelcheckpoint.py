# -*- coding: utf-8 -*-
# @project：wholee_get_other_keyword
# @author:caojinlei
# @file: modelcheckpoint.py
# @time: 2021/06/18
import torch
import numpy as np


class ModelCheckPoint:
    def __init__(self, model, optimizer, args, log, epoch_freq=1, mode='min', save_best_model=True):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.log = log
        self.epoch_freq = epoch_freq
        self.save_best_model = save_best_model
        if mode == 'min':
            self.monitor_op = np.less  # 求最小值
            self.best = np.Inf  # 正无穷
        else:
            self.monitor_op = np.greater  # 求最大值
            self.best = -np.Inf  # 负无穷

    def save_info(self, epoch):
        state = {
            'module': self.args.module,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        resume_path = 'output/checkpoints/{}_first.pth'.format(self.args.module)
        torch.save(state, resume_path)
        return state

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

    def save_step(self, state, current):
        if self.save_best_model:
            best_path = 'output/checkpoints/{}_best.pth'.format(self.args.module)
            if self.monitor_op(current, self.best):
                self.best = current
                state['best'] = self.best
                torch.save(state, best_path)
                self.log.info(f'save the best module where loss is {state["best"]}')
        else:
            if state['epoch'] % self.epoch_freq == 0:
                state['epoch'] += 1
                epoch_path = 'output/checkpoints/{}_checkpoint_epoch{}.pth'.format(self.args.module, state['epoch'])
                torch.save(state, epoch_path)

        return state

