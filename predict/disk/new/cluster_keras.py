import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from data_normalize import *
from data_sampling import *
from abnormal_detect_one_diff import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

# 设置每个文本序列的最大长度，当序列的长度小于maxlen时，将用0来进行填充，当序列的长度大于maxlen时，则进行截断
maxlen = 100


def read_csv(file):
    dateparse = lambda dates: time_set_fun(dates)
    df = pd.read_csv(file)
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

    from sklearn.preprocessing import LabelEncoder
    df['serial_number'] = LabelEncoder().fit_transform(df['serial_number'])
    df['model'] = LabelEncoder().fit_transform(df['model'])

    return df


def lstm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # 设置最大特征的数量，对于文本，就是处理的最大单词数量。若被设置为整数，则被限制为待处理数据集中最常见的max_features个单词
    max_features = 20000
    # 设置训练的轮次
    batch_size = 32

    # 创建网络结构
    model = Sequential()
    model.add(Embedding(len(X_train), 2, input_length=7))
    model.add(LSTM(64))
    # model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # 训练模型
    print('Train...')
    result = model.fit(X_train, y_train, batch_size=batch_size, epochs=4, validation_data=[X_test, y_test])
    print(result)
    val_acc_list = result.history['val_acc']
    val_mean = sum(val_acc_list) / len(val_acc_list)
    print("val_mean:", val_mean)
    print("should==1's predict value:", result.model.predict(X_test)[y_test == 1])
    return val_mean


if __name__ == '__main__':
    input_file = "E:\\mldata\\disknew\\csv\\8t_all.csv";
    df: DataFrame = read_data(input_file)

    np_array = np.array(pd.DataFrame(df,
                                     columns=['failure', 'serial_number', 'model', 'smart_9_raw', 'smart_2_raw',
                                              'smart_12_raw',
                                              'smart_196_raw',
                                              'smart_198_raw']))

    zero_num=1000
    np.random.shuffle(np_array)

    zero_array = np_array[np_array[:, 0] == 0]
    np.random.shuffle(zero_array)
    sampled_zero_data = zero_array[0:zero_num, :]
    print("sampled label=0 data shape: " + str(sampled_zero_data.shape))
    one_array = np_array[np_array[:, 0] == 1]
    sampled_one_data = one_array
    sample_result = np.vstack((sampled_zero_data, sampled_one_data))


    np_smote = smote_enn_data(sample_result)
    # np_smote = np_array
    X = np_smote[:, 1:]
    # X = np_smote[:, [1, 16, 4, 20, 24, 28]]
    # X = np_smote[:, [16,6,3,20,26]]
    # X = np_smote[:, [4, 5:]]
    X = min_max(X)
    y = np_smote[:, 0]

    lstm(X, y)
