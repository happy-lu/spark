# coding=utf-8

from sklearn_cluster.decomposition import PCA
import numpy as np
from data_normalize import *

if __name__ == '__main__':
    file = "E://mldata//disknew//failed//fdata.csv";

    df: DataFrame = read_csv(file)
    df = df.dropna(axis=1, how='all')

    X = np.array(df)
    X = X[:, 5:]

    print(X.shape)

    pca = PCA(n_components=2)
    pca.fit(X)
    result=pca.transform(X)

    print(result.shape)
    print(result)

