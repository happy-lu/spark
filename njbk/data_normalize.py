# coding=utf-8

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def sigmoid(arr):
    s = 1 / (1 + np.exp(-arr))
    # print(s)
    return s


def std(arr):
    s = (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)
    # print(s)
    return s


def min_max(arr):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(arr)


def standard(arr):
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(arr)


def normalize_data(arr):
    # mm_np = min_max(arr)
    # sm_np = sigmoid(arr)
    st_np = standard(arr)

    return st_np

# if __name__ == '__main__':
#     file = "E://mldata//disknew//failed//fdata.csv";
#
#     df: DataFrame = read_csv(file)
#     # print(df.head(3))
#
#     titles = list(df)
#     for i in range(5, len(titles), 2):
#         column_name = titles[i]
#         print(column_name + ":\n")
#         df1 = df[[column_name]]
#         df1=df1.fillna(0)
#         # print(df1.head(3))
#
#         df_np = normalize_to_np(df1)
#         y_pred = KMeans(n_clusters=2, n_init=1).fit_predict(df_np)
#         print(column_name + ":\n" + str(y_pred))
#         #
#         # plt.scatter(df_np[:, 0], df_np[:, 1], c=y_pred)
#         # plt.show()
