import os

from pandas import DataFrame, Series

# os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import matplotlib.pyplot as plt
import pandas as pd
import datetime
from statsmodels.tsa.arima_model import ARIMA


def time_set_fun(dates):
    dt = pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dt = dt.replace(second=0, minute=0)
    return dt


def read_csv(file):
    dateparse = lambda dates: time_set_fun(dates)
    df = pd.read_csv(file, parse_dates=['createtime'], index_col='id', date_parser=dateparse)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})

    return df


def test_parameters(sel_frame, target, params):
    # ARIMA（p，d，q）模型中选择合适模型，其中p为自回归项，d为差分阶数，q为移动平均项数。

    sel_frame = sel_frame.set_index(['createtime'])

    # # 自相关图
    # from statsmodels.graphics.tsaplots import plot_acf
    # plot_acf = plot_acf(sel_frame)
    # plot_acf.show()

    # # 偏自相关图
    # from statsmodels.graphics.tsaplots import plot_pacf
    # plot_pacf = plot_pacf(sel_frame)
    # plot_pacf.show()

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

    # bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            # 存在部分报错，所以用try来跳过报错。
            try:
                tmp.append(ARIMA(sel_frame, (p, params[1], q)).fit().bic)
            except Exception as err:
                print(err)
                tmp.append(None)
        bic_matrix.append(tmp)

    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    # 先用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()

    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))

    plt.show()


def get_real_data_frame(df, target):
    # pandas_frame = df.loc[df['hostaddr'] == '192.168.232.183']
    pandas_frame = df[df['hostaddr'] == '192.168.232.183']
    sel_frame: DataFrame = pandas_frame.filter(items=['createtime', target])

    return sel_frame


def get_predict_increased_data_frame(sel_frame, target, predict_start_day, shift_days, params):
    # 建立ARIMA模型
    test_frame = sel_frame.set_index('createtime')
    test_frame: DataFrame = test_frame.asfreq('1H', method='bfill')
    print("pre_count:" + str(test_frame.count()))

    # test_frame = duplicate_frame(test_frame, predict_start_day, shift_days)
    # print("dup_count:" + str(test_frame.count()))

    model: ARIMA = ARIMA(test_frame, (params[0], params[1], params[2])).fit()
    # 给出一份模型报告
    # print("model summary:\n" + str(model.summary2()))

    en_day = datetime.datetime.strptime(predict_start_day, "%Y-%m-%d") + datetime.timedelta(
        days=shift_days) + datetime.timedelta(seconds=-1)
    times = pd.date_range(predict_start_day, en_day, freq='h')
    series = Series(model.forecast(shift_days * 24)[0], index=times)
    predict_dta: DataFrame = Series.to_frame(series)
    predict_dta.columns = ['predict_' + target]
    print("predict_dta:" + str(
        model.predict(start=predict_start_day, end=en_day + datetime.timedelta(seconds=1)).head(10)))
    print("forecast_dta:" + str(model.forecast(100)))

    max_value = test_frame.max()[0]
    min_value = test_frame.min()[0]
    predict_dta = predict_dta['predict_' + target].map(lambda a: a - max_value + min_value if a >= max_value else a)

    return predict_dta


def get_predict_increased_data_frame2(sel_frame, target, predict_start_day, shift_days, params):
    # 建立ARIMA模型
    test_frame = sel_frame.set_index('createtime')
    test_frame: DataFrame = test_frame.asfreq('1H', method='bfill')
    print("pre_count:" + str(test_frame.count()))

    print("dup_count:" + str(test_frame.count()))

    model: ARIMA = ARIMA(test_frame, (params[0], params[1], params[2])).fit()
    # 给出一份模型报告
    # print("model summary:\n" + str(model.summary2()))

    # 复制最后一个数据，并加上ARIMA预测出的结果
    en_day = datetime.datetime.strptime(predict_start_day, "%Y-%m-%d") + datetime.timedelta(days=shift_days)
    predict_dta: DataFrame = Series.to_frame(model.predict(start=predict_start_day, end=en_day))
    predict_dta.columns = ['predict_' + target]
    print("predict_dta:" + str(predict_dta.head(10)))
    predict_dta = predict_dta.cumsum()
    print("predict_dta_cumsum:" + str(predict_dta.head(10)))

    last_value = test_frame.tail(1)
    final_predict_dta = predict_dta + last_value.values[0][0]

    print("final_predict_dta:" + str(final_predict_dta.head(10)))

    return final_predict_dta


def get_predict_data_frame(sel_frame, target, predict_start_day, shift_days, params):
    # 建立ARIMA模型
    test_frame = sel_frame.set_index('createtime')
    test_frame: DataFrame = test_frame.asfreq('1H', method='bfill')
    print("pre_count:" + str(test_frame.count()))

    test_frame = duplicate_frame(test_frame, predict_start_day, shift_days)
    print("dup_count:" + str(test_frame.count()))

    model: ARIMA = ARIMA(test_frame, (params[0], params[1], params[2])).fit()
    # 给出一份模型报告
    # print("model summary:\n" + str(model.summary2()))

    en_day = datetime.datetime.strptime(predict_start_day, "%Y-%m-%d") + datetime.timedelta(days=shift_days)
    predict_dta: DataFrame = Series.to_frame(model.predict(start=predict_start_day, end=en_day))
    predict_dta.columns = ['predict_' + target]
    print("predict_dta:" + str(predict_dta.head(10)))
    print("forecast_dta:" + str(model.forecast(100)))

    # en_day = datetime.datetime.strptime(predict_start_day, "%Y-%m-%d") + datetime.timedelta(
    #     days=shift_days) + datetime.timedelta(seconds=-1)
    # times = pd.date_range(predict_start_day, en_day, freq='h')
    # series = Series(model.forecast(shift_days * 24)[0], index=times)
    # predict_dta: DataFrame = Series.to_frame(series)
    # predict_dta.columns = ['predict_' + target]
    # print("predict_dta:" + str(
    #     model.predict(start=predict_start_day, end=en_day + datetime.timedelta(seconds=1)).head(10)))
    # print("forecast_dta:" + str(model.forecast(100)))

    return predict_dta


def duplicate_frame(df, predict_start_day, shift_days):
    end_date = datetime.datetime.strptime(predict_start_day, "%Y-%m-%d") + datetime.timedelta(seconds=-1)
    oneday_average = df[:end_date]
    oneday_average_dup = oneday_average.shift(shift_days, freq='d')
    oneday_average_dup = oneday_average_dup[predict_start_day:]
    oneday_average = oneday_average.append(oneday_average_dup)

    return oneday_average


if __name__ == '__main__':
    file = "E://mldata//predict//hostresource_nonet.csv";
    target = 'mem_usage'

    predict_start_day = '2018-07-17'
    shift_days = 28

    params = {'mem_usage': [1, 1, 2], 'cpu_usage': [1, 0, 1]}
    # is_cal_param = True;
    is_cal_param = False;

    df = read_csv(file)
    print(df.head(3))
    sel_frame = get_real_data_frame(df, target)
    print(sel_frame.count())
    part_data_frame = sel_frame[sel_frame['createtime'] < predict_start_day]
    print(part_data_frame)

    # 计算出应该用什么参数去预测
    if is_cal_param:
        test_parameters(part_data_frame, target, params[target])
        os._exit(0)

    # 预测的数据
    predict_method = get_predict_increased_data_frame if target == 'mem_usage' else get_predict_data_frame
    predict_data_frame = predict_method(part_data_frame, target, predict_start_day, shift_days, params[target])
    plt.rcParams['axes.unicode_minus'] = False

    # 曲线图
    sel_frame = sel_frame.set_index(['createtime'])

    # 移动平均线
    oneday_average = sel_frame.rolling(24).mean()
    oneday_average.columns = ['mean_' + target]
    oneday_average = duplicate_frame(oneday_average, predict_start_day, shift_days)

    combine_frame = sel_frame.join(predict_data_frame, how='outer').join(oneday_average, how='left')

    # predict_data_frame2 = get_predict_increased_data_frame2(part_data_frame, target, predict_start_day, shift_days,
    #                                                         params[target])
    # predict_data_frame2.columns = ['predict2_' + target]
    # combine_frame = combine_frame.join(predict_data_frame2, how='left')

    combine_frame.plot()

    plt.show()
