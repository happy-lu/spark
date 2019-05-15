import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from data_normalize import *
from abnormal_detect_one_diff import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

# 设置最大特征的数量，对于文本，就是处理的最大单词数量。若被设置为整数，则被限制为待处理数据集中最常见的max_features个单词
max_features = 20000
# 设置每个文本序列的最大长度，当序列的长度小于maxlen时，将用0来进行填充，当序列的长度大于maxlen时，则进行截断
maxlen = 100
# 设置训练的轮次
batch_size = 32


def time_set_fun(dates):
    dt = pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dt = dt.replace(second=0, minute=0)
    return dt


def read_csv(file):
    dateparse = lambda dates: time_set_fun(dates)
    # df = pd.read_csv(file, parse_dates=['date'], index_col='date', date_parser=dateparse)
    df = pd.read_csv(file, parse_dates=['date'], date_parser=dateparse)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})

    return df


def lstm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # 创建网络结构
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=31))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # 训练模型
    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=4, validation_data=[X_test, y_test])


if __name__ == '__main__':
    input_file = "E:\\mldata\\disknew\\csv\\time_diff.csv";
    df: DataFrame = read_data(input_file)

    np_array = np.array(df)
    std_np_array = min_max(np_array)
    lstm(std_np_array[:, 1:], std_np_array[:, 0])
