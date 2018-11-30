# -*-coding:utf-8-*-

import math

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


class RankObj():
    weight = 0
    reason = {}

    def __str__(self):
        return str(self.weight)


class ItemBasedCF:
    def __init__(self, train_file):
        self.train_file = train_file
        self.readData()

    def readData(self):
        # 读取文件，并生成用户-物品的评分表和测试集
        self.train = dict()  # 用户-物品的评分表
        i = 0
        for line in open(self.train_file):
            if i == 0:
                i += 1
                continue
            # user,item,score = line.strip().split(",")
            user, item, score = line.strip().split(",")
            self.train.setdefault(user, {})
            self.train[user][item] = int(float(score))

    def item_similarity(self):
        # 建立物品-物品的共现矩阵
        C = dict()
        # 物品被多少个不同用户购买
        N = dict()
        for user, items in self.train.items():
            for i in items.keys():
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items.keys():
                    # if i == j:
                    #     continue
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        # 计算相似度矩阵
        self.W = dict()
        for i, related_items in C.items():
            self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))

        # 缺少对W在j轴的归一化步骤
        return self.W

    # 给用户user推荐，前K个相关物品的前N个关联物品
    def recommend(self, user, K=70, N=100):
        rank = dict()
        action_item = self.train[user]  # 用户user产生过行为的item和评分

        for item, score in action_item.items():
            for j, wj in sorted(self.W[item].items(), key=lambda x: x[1], reverse=True)[0:K]:
                # if j in action_item.keys():
                #     continue
                rank.setdefault(j, RankObj())
                rank[j].weight += score * wj
                rank[j].reason[item] = score * wj

        remain_dict = rank
        if len(remain_dict) > 0:
            sorted_dcit = sorted(remain_dict.items(), key=lambda x: x[1].weight, reverse=True)
            if len(sorted_dcit) > N:
                return dict(sorted_dcit[0:N])
            else:
                return dict(sorted_dcit)
        return remain_dict

    def precision_recall(self, test, k=10, n=100, percent=0.5):
        hit = 0
        n_recall = 0
        n_precision = 0

        user_list, test_list, recommend_list = [], [], []
        for test_user, test_items in test.train.items():
            if test_user not in self.train:
                continue

            test_buy_dcit = self.get_test_fit_item_dict(test_items, 10)
            # 不预测test集里没有购买记录的
            if len(test_buy_dcit) == 0:
                continue

            recommend_items = self.recommend(test_user, k, n)
            if len(recommend_items) == 0:
                recommend_buy_dcit = {}
            else:
                recommend_buy_dcit = self.get_recommend_fit_item_dict(recommend_items, 10, percent)

            if len(test_buy_dcit) == len(recommend_buy_dcit) == 0:
                continue

            # if len(test_buy_dcit) > 0:
            #     print(test_buy_dcit)
            #     print(str_rankObj(recommend_items))

            cur_fit = len(test_buy_dcit.keys() & recommend_buy_dcit.keys())
            cur_precision = (cur_fit / len(recommend_buy_dcit)) if len(recommend_buy_dcit) > 0 else 0
            cur_recall = (cur_fit / len(test_buy_dcit)) if len(
                test_buy_dcit) > 0 else 0

            hit += cur_fit
            n_recall += len(test_buy_dcit)
            n_precision += len(recommend_buy_dcit)

            user_list.append([test_user, cur_precision, cur_recall])
            test_list.append(test_buy_dcit)
            recommend_list.append(recommend_buy_dcit)

        return [hit / (1.0 * n_precision), hit / (1.0 * n_recall)], user_list, test_list, recommend_list

    def get_test_fit_item_dict(self, recommend_items, score):
        return dict(filter(lambda x: x[1] == score, recommend_items.items()))

    def get_recommend_fit_item_dict(self, recommend_items, score, percent=1):
        return dict(filter(lambda x: x[1].weight > score * percent, recommend_items.items()))


def str_rankObj(obj):
    str1 = ""
    for k, v in obj.items():
        str1 += k + "(" + str(v) + "), "
    return str1


def show_as_line(rdd, legend_desc):
    df = DataFrame(np.random.randint(0, 150, size=(5, 4)),
                   columns=['python', 'java', 'php', 'ruby'],
                   index=list('abcde'))
    df.plot()


if __name__ == '__main__':
    # 声明一个ItemBased推荐的对象
    k = 100
    n = 10
    percent = 0.5
    item_cf = ItemBasedCF("data//event_tagged_train.csv")
    item_cf.item_similarity()
    # recommedDic = item_cf.recommend("57905")
    # for k, v in recommedDic.items():
    #     print(k, "\t", v)

    test = ItemBasedCF("data//event_tagged_test.csv")
    result, user_list, train_list, test_list = item_cf.precision_recall(test, k, n, percent)
    print("test result(precision,recall):" + str(result))

    # for i, k in enumerate(user_list):
    #     print("user:" + str(k) + ",test_list:" + str(train_list[i]) + ",recommend_list:" + str_rankObj(test_list[i]))
