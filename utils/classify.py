# -*- coding: utf-8 -*-
# @project：wholee_get_walks
# @author:caojinlei
# @file: classify.py
# @time: 2021/05/28
import logging
import heapq
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import numpy
from tqdm import tqdm
from utils.Logginger import init_logger

logger = init_logger("metrics", logging_path='output/')


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(np.array(X)))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class NodeClassify:
    def __init__(self, X, Y, embedding, estimator=LogisticRegression(), test_size=0.2, seed=0):
        self.X = X
        self.Y = Y
        self.embedding = embedding
        self.clf = TopKRanker(estimator)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)
        self.test_size = test_size
        self.seed = seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                                random_state=666)

    def train(self):
        self.binarizer.fit(self.Y)
        x_train_emb = [self.embedding[x] for x in self.X_train]
        y_train = self.binarizer.transform(self.y_train)
        self.clf.fit(x_train_emb, y_train)

    def evaluate(self):
        top_k_list = [len(l) for l in self.y_test]
        x_test_emb = [self.embedding[x] for x in self.X_test]
        y_ = self.clf.predict(x_test_emb, top_k_list=top_k_list)
        y_test = self.binarizer.transform(self.y_test)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(y_test, y_, average=average)
            logger.info(f'{average}: {results[average]}\n')
        # results['acc'] = accuracy_score(y_test, y_)
        # logging.info(f'acc: {results["acc"]}')
        return results




def cos_sim(x, y):
    x_norm = np.linalg.norm(np.array(x))
    y_norm = np.linalg.norm(np.array(y))
    result = np.dot(x, y) / (x_norm * y_norm)
    return result


def precision_link(pos_sim_list, neg_sim_list, topk_list: list):
    """

    :param pos_sim_list:
    :param neg_sim_list:
    :param topk_list:
    :return:
    """
    logger.info('Start calculate precision')
    tar_pos_list = [1 for x in range(len(pos_sim_list))]
    tag_neg_list = [0 for x in range(len(neg_sim_list))]
    all_sim_list = np.r_[pos_sim_list, neg_sim_list]
    all_tag_list = np.r_[tar_pos_list, tag_neg_list]
    all_sim_sort = np.argsort(all_sim_list)
    pre_dict = {}
    for topk in topk_list:
        top_index_list = all_sim_sort[-topk:]
        pos_count = 0
        for i in top_index_list:
            if all_tag_list[i] == 1:
                pos_count += 1
        pre_key = f'pre_score_{str(topk)}'
        pre_score = pos_count / topk
        pre_dict[pre_key] = pre_score

    return pre_dict


def auc(pos_sim_list, neg_sim_list):
    """
    参考2011年吕琳媛《Link prediction in complex networks: A survey》
    AUC = (n1+0.5n2)/n*m
    :param pos_sim_list:
    :param neg_sim_list:
    :return:
    """
    n = len(pos_sim_list)
    m = len(neg_sim_list)
    n1 = 0
    n2 = 0
    logger.info('Start calculate auc')
    j = 0
    for i in tqdm(range(n)):
        for x in range(j, m):
            pos_sim = pos_sim_list[i]
            neg_sim = neg_sim_list[x]
            if pos_sim > neg_sim:
                n1 += m - x
                j = x
                break
            elif pos_sim == neg_sim:
                n2 += 1
                continue
    auc_score = (n1 + 0.5 * n2) / (n * m)
    logger.info('Finish calculate auc')
    return auc_score


class LinkPredict:
    # TODO做一个二分类
    def __init__(self, pos_list, neg_list, embedding):
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.embedding = embedding

    def train(self):
        length = len(self.pos_list)
        pos_sim_list = []
        neg_sim_list = []
        for i in tqdm(range(length)):
            pos_pre = self.embedding[self.pos_list[i][0]]
            pos_lat = self.embedding[self.pos_list[i][1]]
            pos_sim = cos_sim(pos_pre, pos_lat)
            pos_sim_list.append(pos_sim)

            neg_pre = self.embedding[self.neg_list[i][0]]
            neg_lat = self.embedding[self.neg_list[i][1]]
            neg_sim = cos_sim(neg_pre, neg_lat)
            neg_sim_list.append(neg_sim)
        return pos_sim_list, neg_sim_list

    def evaluate(self, pos_sim_list, neg_sim_list, topk_list):
        score_result = precision_link(pos_sim_list, neg_sim_list, topk_list)
        auc_score = auc(pos_sim_list, neg_sim_list)
        score_result['auc_score'] = auc_score
        return score_result


class MultiClassifier(object):
    '''
    learn from:
    https://github.com/shenweichen/GraphEmbedding/blob/master/ge/classify.py
    '''

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer()

    def fit(self, X, y, y_all):
        '''
        :param X:
        :param y:
        :param y_all: 所有的标签
        :return:
        '''
        self.binarizer.fit(y_all)
        X_train = [self.embeddings[x] for x in X]
        y_train = self.binarizer.transform(y)
        self.clf.fit(X_train, y_train)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        y_pred = self.clf.predict(X_, top_k_list=top_k_list)
        return y_pred

    def evaluate(self, X, y):
        top_k_list = [len(l) for l in y]
        y_pred = self.predict(X, top_k_list)
        y = self.binarizer.transform(y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(y, y_pred, average=average)
        results['acc'] = accuracy_score(y, y_pred)
        print('-------------------')
        print(results)
        print('-------------------')
        return results

    def evaluate_hold_out(self, X, y, test_size=0.2, random_state=123):
        np.random.seed(random_state)
        train_size = int((1-test_size) * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(train_size)]
        y_train = [y[shuffle_indices[i]] for i in range(train_size)]
        X_test = [X[shuffle_indices[i]] for i in range(train_size, len(X))]
        y_test = [y[shuffle_indices[i]] for i in range(train_size, len(X))]

        self.fit(X_train, y_train, y)

        return self.evaluate(X_test, y_test)

