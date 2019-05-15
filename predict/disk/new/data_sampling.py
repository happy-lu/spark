# coding=utf-8

import os
import numpy as np
import pandas as pd
from pandas import DataFrame

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
# np.set_printoptions(threshold=np.inf)
# 若想不以科学计数显示:
np.set_printoptions(suppress=True)


def time_set_fun(dates):
    dt = pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dt = dt.replace(second=0, minute=0)
    return dt


def read_csv(file):
    dateparse = lambda dates: time_set_fun(dates)
    df = pd.read_csv(file, parse_dates=['date'], index_col='date', date_parser=dateparse)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})

    return df


def under_sample(X, y):
    pass
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_sample(X, y)
    # # print(X_resampled.shape)
    # # print(y_resampled.shape)
    # # print(y_resampled[y_resampled == 1].shape)
    #
    # y_resampled = y_resampled.reshape(y_resampled.shape[0], 1)


def laod_data(output_file):
    return np.loadtxt(output_file, delimiter=',')

def smote_enn_data(sample_result):
    print("before SMOTE data shape:" + str(sample_result.shape))

    X_resampled_smote, y_resampled_smote = SMOTEENN().fit_sample(sample_result[:, 1:], sample_result[:, 0])
    y_resampled_smote = y_resampled_smote.reshape(y_resampled_smote.shape[0], 1)
    result = np.hstack((y_resampled_smote, X_resampled_smote))
    print("after SMOTE data shape:" + str(result.shape))

    return result


def smote_data(sample_result):
    print("before SMOTE data shape:" + str(sample_result.shape))

    X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(sample_result[:, 1:], sample_result[:, 0])
    y_resampled_smote = y_resampled_smote.reshape(y_resampled_smote.shape[0], 1)
    result = np.hstack((y_resampled_smote, X_resampled_smote))
    print("after SMOTE data shape:" + str(result.shape))

    return result


def sample_data(input_file, zero_data_num):
    df: DataFrame = read_csv(input_file)
    origin_cols = set(df.columns)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')
    drop_cols = origin_cols.difference(set(df.columns))
    # list(drop_cols).sort()
    print("drop columns: " + str(drop_cols))
    print("after drop columns: " + str(list(df.columns)))
    # df = df.fillna(0)
    all_data = np.array(df)
    X = all_data[:, 4:]
    y = all_data[:, 3]
    print("origin data shape: " + str(X.shape))
    print("origin label=1 data shape: " + str(y[y == 1].shape))
    zero_array = all_data[all_data[:, 3] == 0]
    np.random.shuffle(zero_array)
    sampled_zero_data = zero_array[0:zero_data_num, 3:]
    print("sampled label=0 data shape: " + str(sampled_zero_data.shape))
    one_array = all_data[all_data[:, 3] == 1]
    sampled_one_data = one_array[:, 3:]
    sample_result = np.vstack((sampled_zero_data, sampled_one_data))
    print("sample_data data shape:" + str(sample_result.shape))
    return sample_result


if __name__ == '__main__':
    input_file = "E:\\mldata\\disknew\\csv\\single.csv";
    sample_file = "E:\\mldata\\disknew\\csv\\single_sample.csv";
    data_num = 10000

    if os.path.exists(sample_file):
        np_array = laod_data(sample_file)
    else:
        np_array = sample_data(input_file, data_num)
        np.savetxt(sample_file, np_array, fmt="%d", delimiter=",")

    smote_np = smote_data(np_array)
    print("data shape:" + str(smote_np.shape))
