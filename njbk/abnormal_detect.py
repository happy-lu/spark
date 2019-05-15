import os
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.ensemble import IsolationForest
from sklearn import svm

import matplotlib.pyplot as plt
import matplotlib as mpl

from data_normalize import *


def read_csv(file):
    # df = pd.read_csv(file, parse_dates=['date'], index_col='date', date_parser=dateparse)
    df = pd.read_csv(file,header=None)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})

    return df


def read_data(input_file):
    df: DataFrame = read_csv(input_file)
    origin_cols = set(df.columns)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')
    drop_cols = origin_cols.difference(set(df.columns))
    # list(drop_cols).sort()
    print("drop columns: " + str(drop_cols))
    print("after drop columns: " + str(list(df.columns)))

    # df = pd.get_dummies(df, columns=[3])
    df.drop([3], axis=1, inplace=True)
    return df


def one_diff_exam(df):
    # 使用指标一阶差分进行异常检测
    data = np.array(df)
    data1 = data.copy()
    data2 = data.copy()
    data1 = np.delete(data1, 0, 0)
    data2 = np.delete(data2, -1, 0)
    data = data1 - data2

    df3 = pd.DataFrame(data)

    # 删除包含0的列
    df_rm0 = df3.ix[:, ~((df3 == 0).all())]
    data = np.array(df_rm0)

    # 使用oneclasssvm
    # algorithm = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.004)
    algorithm = IsolationForest(n_estimators=1000)
    model = algorithm.fit(data[:-100])
    pre_y = model.predict(data[-100:])
    print(pre_y)

    # 异常检测结果画图
    # df1 = df[['failure','smart_10_raw']][1:]
    df1 = df_rm0[-100:]
    df1['failure'] = pre_y

    df2 = df1[df1['failure'] == 1]
    df3 = df1[df1['failure'] == -1]

    print(df3.index)
    # plt.figure()
    df2.plot()
    df3.plot()
    plt.show()



if __name__ == '__main__':
    input_file = "E:\\mldata\\njbk\\smp_cpu.csv";
    df: DataFrame = read_data(input_file)

    sdf = standard(np.array(df))
    one_diff_exam(sdf)
