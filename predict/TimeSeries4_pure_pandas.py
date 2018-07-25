import os
from typing import Any, Union

from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
from statsmodels.tsa.arima_model import ARIMA
import numpy as np


def time_set_fun(dates):
    dt = pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dt = dt.replace(second=0, minute=0)
    return dt


def read_csv(file):
    dateparse = lambda dates: time_set_fun(dates)
    df = pd.read_csv(file, parse_dates=['createtime'], index_col='id', date_parser=dateparse)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})

    return df


def get_real_data_frame(df, target):
    # pandas_frame = df.loc[df['hostaddr'] == '192.168.232.183']
    pandas_frame = df[df['hostaddr'] == '192.168.232.183']
    sel_frame: DataFrame = pandas_frame.filter(items=['createtime', target])

    # sel_frame = pandas_frame[['createtime', target]]
    # sel_frame = pd.DataFrame(pandas_frame, columns=('createtime', target))
    return sel_frame


def get_predict_data_frame(sel_frame, target):
    # 建立ARIMA(1,0 , 1)模型
    # sel_frame['createtime'] = pd.to_datetime(sel_frame['createtime'])

    test_frame = sel_frame.set_index('createtime')
    test_frame = test_frame.asfreq('1H', method='bfill')
    # test_frame.index._freq='1H'
    # test_frame.index.asfreq('1H')
    # test_frame = test_frame.dropna(axis=0, how='any')
    # print(test_frame.head(10000))

    model = ARIMA(test_frame, (1, 0, 1)).fit()
    # 给出一份模型报告
    # print("model summary:\n" + str(model.summary2()))
    # print("forecast data:\n" + str(model.forecast(5)))

    # predict_dta = model.forecast(2000)
    # print(predict_dta[0])
    # plt.plot(predict_dta[0], color='red')

    predict_dta: DataFrame = Series.to_frame(model.predict(start="2018-07-10 00:00:00", end="2018-09-20 00:00:00"))
    predict_dta.columns = ['predict_' + target]
    print("predict_dta:" + str(predict_dta.head(10)))

    # fig = model.plot_predict(start="2018-07-10 00:00:00", end="2018-07-20 00:00:00")
    # plt.plot(model.fittedvalues, color='red')
    return predict_dta

    # plt.show()


def test_parameters(sel_frame, target):
    # ARIMA（p，d，q）模型中选择合适模型，其中p为自回归项，d为差分阶数，q为移动平均项数。

    # # 自相关图
    # from statsmodels.graphics.tsaplots import plot_acf
    # sel_frame = sel_frame.set_index(['createtime'])
    # plot_acf = plot_acf(sel_frame)
    # plot_acf.show()

    test_data = sel_frame[target]
    # 平稳性检测
    from statsmodels.tsa.stattools import adfuller as ADF
    # print(sel_frame['createtime'].tolist())l
    print(u'原始序列的ADF检验结果为（第一个返回值为adf，若小于1%5%10%均值则为平稳序列，d=0）：', ADF(test_data))

    from statsmodels.stats.diagnostic import acorr_ljungbox
    # 返回统计量和p值
    print(u'差分序列的白噪声检验结果为(p值)：', acorr_ljungbox(test_data, lags=1))

    # ARIMA，计算p和q
    # 一般阶数不超过length/10
    pmax = int(len(test_data) / 100)
    qmax = int(len(test_data) / 100)
    test_frame = sel_frame.set_index(['createtime'])

    # bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            # 存在部分报错，所以用try来跳过报错。
            try:
                tmp.append(ARIMA(test_frame, (p, 0, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    # 先用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()

    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))

    plt.show()


if __name__ == '__main__':
    file = "E://mldata//predict//hostresource_nonet.csv";
    target = 'cpu_usage'
    # is_cal_param = True;
    is_cal_param = False;

    df = read_csv(file)
    print(df.head(3))
    sel_frame = get_real_data_frame(df, target)
    print(sel_frame.count())
    part_data_frame = sel_frame[sel_frame['createtime'] <= '2018-07-10']
    print(part_data_frame)

    if is_cal_param:
        test_parameters(part_data_frame, target)
        os._exit(0)

    predict_data_frame = get_predict_data_frame(part_data_frame, target)
    plt.rcParams['axes.unicode_minus'] = False

    # 曲线图
    sel_frame = sel_frame.set_index(['createtime'])

    combine_frame = sel_frame.join(predict_data_frame, how='outer')
    combine_frame.plot()

    # fig, ax = plt.subplots()
    # yticks = range(0, 30, 2)
    # ax.set_yticks(yticks)
    plt.show()
