import os

os.environ["PYSPARK_PYTHON"] = "D://Python3//python.exe"

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    file = "E://mldata//predict//hostresource_nonet.csv";

    # d = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pd.DataFrame(data=d)

    # # 用pandas将时间转为标准格式
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    # # 将时间栏合并,并转为标准时间格式
    # rawdata = pd.read_csv('RealMarketPriceDataPT.csv', parse_dates={'timeline': ['date', '(UTC)']},
    #                       date_parser=dateparse)

    df = pd.read_csv(file, parse_dates=['createtime'], index_col='createtime', date_parser=pd.to_datetime)
    # df = pd.read_csv(file, dtype={'createtime': pd.datetime64})
    print(df.head(3))
    pandas_frame = df[(True == df['hostaddr'].isin(['192.168.232.183']))]
    sel_frame = pandas_frame[[ 'cpu_usage']]
    # sel_frame = pd.DataFrame(pandas_frame, columns=('createtime', 'cpu_usage'))
    # sel_frame.set_index('createtime')
    # sel_frame.reindex(['createtime'])
    # sel_frame['createtime'] = dateparse(sel_frame['createtime'])

    print(sel_frame.count())
    print(sel_frame.dtypes)

    plt.rcParams['axes.unicode_minus'] = False
    sel_frame.plot()
    plt.show()
