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
        # return str(self.weight) + "," + str(self.reason)


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
            user, item, score, time = line.strip().split(",")
            self.train.setdefault(user, {})
            self.train[user].setdefault(item, 0)
            self.train[user][item] += int(float(score))

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
                    # C[i][j] += 1
                    C[i][j] += 1

        # # # 归一化步骤
        # C = normalize(C)

        # 计算相似度矩阵
        self.W = dict()
        for i, related_items in C.items():
            self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))

        return self.W

    # 给用户user推荐，前K个相关物品的前N个关联物品
    def recommend(self, user, K=70, N=100, percent=0.1):
        rank = dict()
        train_item = self.train[user]  # 用户user产生过行为的item和评分

        for item, item_score in train_item.items():
            try:
                rank.setdefault(item, RankObj())
                rank[item].weight = +item_score
                rank[item].reason[item] = item_score

                for j, wj in sorted(self.W[item].items(), key=lambda x: x[1], reverse=True)[0:K]:
                    rank.setdefault(j, RankObj())

                    if j in train_item.keys():
                        continue

                    point = wj * item_score * percent
                    rank[j].weight += point
                    rank[j].reason[item] = point
            except Exception as err:
                print(err)

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

            # test_buy_dcit = self.get_test_fit_item_dict(test_items, 10)
            test_buy_dcit = test_items

            if len(test_buy_dcit) == 0:
                continue

            recommend_items = self.recommend(test_user, k, n, percent)
            recommend_buy_dcit = recommend_items
            # if len(recommend_items) == 0:
            #     recommend_buy_dcit = {}
            # else:
            #     recommend_buy_dcit = self.get_recommend_fit_item_dict(recommend_items, 10)

            # if len(test_buy_dcit) > 0:
            #     print(test_buy_dcit)
            #     print(str_rankObj(recommend_items))

            tp = len(test_buy_dcit.keys() & recommend_items.keys())
            cur_precision = (tp / len(recommend_buy_dcit)) if len(recommend_buy_dcit) > 0 else 0
            cur_recall = (tp / len(test_buy_dcit)) if len(
                test_buy_dcit) > 0 else 0

            hit += tp
            n_precision += len(recommend_buy_dcit)
            n_recall += len(test_buy_dcit)

            user_list.append([test_user, cur_precision, cur_recall])
            test_list.append(test_buy_dcit)
            recommend_list.append(recommend_buy_dcit)

        return [hit / (1.0 * n_precision), hit / (1.0 * n_recall)], user_list, test_list, recommend_list

    # def get_test_fit_item_dict(self, items, score):
    #     return dict(filter(lambda x: x[1] == score, items.items()))

    # def get_recommend_fit_item_dict(self, items, score, percent=1):
    #     return dict(filter(lambda x: x[1].weight > score * percent, items.items()))


def normalize(target):
    if len(target) == 0:
        return target

    result = dict()
    # 由于这里是对称矩阵，所以根据列归一化可以变成根据行归一化后再转置此矩阵
    for item, v_dict in target.items():
        np_array = np.array(list(v_dict.values()))
        if len(np_array) == 0:
            continue
        normalize = np_array / np.max(np_array, axis=0)

        i = 0
        for k, v in v_dict.items():
            #  转置矩阵
            result.setdefault(k, {})
            result[k].setdefault(item, 0)
            result[k][item] = normalize[i]
            i += 1
    return result


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


def cross_percent():
    # cur best is 0.1
    # 声明一个ItemBased推荐的对象
    k = 35
    n = 3
    percent_base = 0.1
    percent_rate = 0.02
    result_dcit = {}

    for i in range(10):
        percent = percent_base + i * percent_rate
        item_cf = ItemBasedCF("data//event_tagged_train.csv")
        item_cf.item_similarity()

        test = ItemBasedCF("data//event_tagged_cross.csv")
        result, user_list, train_list, cross_list = item_cf.precision_recall(test, k, n, percent)
        result_dcit[percent] = result
        print("test result(precision,recall):" + str(percent) + " " + str(result))

    result_pd = pd.DataFrame.from_dict(result_dcit, orient='index', columns=['precision', 'recall'])
    result_pd.plot()
    plt.show()


def cross_k():
    # cur best is 35
    # 声明一个ItemBased推荐的对象
    k_base = 5
    k_rate = 5
    n = 3
    percent = 0.2
    result_dcit = {}

    for i in range(20):
        k = k_base + i * k_rate
        item_cf = ItemBasedCF("data//event_tagged_train.csv")
        item_cf.item_similarity()

        test = ItemBasedCF("data//event_tagged_cross.csv")
        result, user_list, train_list, cross_list = item_cf.precision_recall(test, k, n, percent)
        result_dcit[k] = result
        print("test result(precision,recall):" + str(k) + " " + str(result))

    result_pd = pd.DataFrame.from_dict(result_dcit, orient='index', columns=['precision', 'recall'])
    result_pd.plot()
    plt.show()


def cross_n():
    # cur best is 1
    # 声明一个ItemBased推荐的对象
    k = 10
    n_base = 1
    n_rate = 5
    percent = 0.2
    result_dcit = {}

    for i in range(10):
        n = n_base + i * n_rate
        item_cf = ItemBasedCF("data//event_tagged_train.csv")
        item_cf.item_similarity()

        test = ItemBasedCF("data//event_tagged_cross.csv")
        result, user_list, train_list, cross_list = item_cf.precision_recall(test, k, n, percent)
        result_dcit[n] = result
        print("test result(precision,recall):" + str(n) + " " + str(result))

    result_pd = pd.DataFrame.from_dict(result_dcit, orient='index', columns=['precision', 'recall'])
    result_pd.plot()
    plt.show()


def test_n():
    # cur best is 1
    # 声明一个ItemBased推荐的对象
    k = 10
    n_base = 1
    n_rate = 5
    percent = 0.2
    result_dcit = {}

    for i in range(10):
        n = n_base + i * n_rate
        item_cf = ItemBasedCF("data//event_tagged_train.csv")
        item_cf.item_similarity()

        test = ItemBasedCF("data//event_tagged_test.csv")
        result, user_list, train_list, cross_list = item_cf.precision_recall(test, k, n, percent)
        result_dcit[n] = result
        print("test result(precision,recall):" + str(n) + " " + str(result))

    result_pd = pd.DataFrame.from_dict(result_dcit, orient='index', columns=['precision', 'recall'])
    result_pd.plot()
    plt.show()


def cross_normalize():
    a = {'a': {'a': 1, "b": 3, "d": 4}, "b": {"a": 3}, "d": {"a": 4, "d": 1}}
    b = normalize(a)
    print(b)


def cross_detail():
    k = 35
    n = 10
    percent = 0.19

    item_cf = ItemBasedCF("data//event_tagged_train.csv")
    item_cf.item_similarity()

    test = ItemBasedCF("data//event_tagged_cross.csv")
    result, user_list, train_list, cross_list = item_cf.precision_recall(test, k, n, percent)
    print("test result(precision,recall):" + str(percent) + " " + str(result))

    for i, k in enumerate(user_list):
        print("user:" + str(k) + ",cross_list:" + str(train_list[i]) + ",recommend_list:" + str_rankObj(cross_list[i]))
        # if i == 10:
        #     break


if __name__ == '__main__':
    # cross_percent()
    # cross_k()
    # cross_n()
    # cross_normalize()
    cross_detail()
    # test_n()
