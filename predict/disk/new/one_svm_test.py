# coding=utf-8
from data_sampling import *
from data_normalize import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import IsolationForest
from sklearn import svm

import copy
import json
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_regress(X, y, reg):
    print("use reg:" + str(reg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # if str(type(reg)) == "<class 'sklearn.svm.classes.OneClassSVM'>":
    #     print("OneClassSVM, need remove failure==1 data")
    #     filter = y_train == 0
    #     X_train = X_train[filter]
    #     y_train = y_train[filter]
    #     print("OneClassSVM, new data shape is", X_train.shape, y_train.shape)

    reg.fit(X_train)

    if hasattr(reg, "intercept_"):
        print(reg.intercept_)
    if hasattr(reg, "coef_"):
        print(reg.coef_)
    if hasattr(reg, "feature_importances_"):
        names = ["smart_1_normalized", "smart_1_raw", "smart_2_normalized", "smart_2_raw", "smart_3_normalized",
                 "smart_3_raw", "smart_4_normalized", "smart_4_raw", "smart_5_normalized", "smart_5_raw",
                 "smart_7_normalized", "smart_7_raw", "smart_8_normalized", "smart_8_raw", "smart_9_normalized",
                 "smart_9_raw", "smart_10_normalized", "smart_10_raw", "smart_12_normalized", "smart_12_raw",
                 "smart_194_normalized", "smart_194_raw", "smart_196_normalized", "smart_196_raw",
                 "smart_197_normalized", "smart_197_raw", "smart_198_normalized", "smart_198_raw",
                 "smart_199_normalized", "smart_199_raw"
                 ]
        print(sorted(zip(map(lambda x: round(x, 4), reg.feature_importances_), names), reverse=True))

    # 模型拟合测试集
    # X_test = [0, 1, 2, 3, 4]
    # X_test = np.row_stack((X_test, [100, 200, 300, 400, 500]))
    # X_test = np.row_stack((X_test, [888, 999, 300, 999, 999]))
    # X_test = np.row_stack((X_test, [10, 20, 30, 40, 50]))
    # X_test = np.row_stack((X_test, [1, 1, 1, 1, 1]))
    # X_test = np.row_stack((X_test, [4, 3, 2, 1, 0]))
    y_pred = reg.predict(X_test)

    if hasattr(reg, "score"):
        print("predict accuracy：", reg.score(X_test, y_test))

    # grid_search(X, y, y_pred, y_test)
    valid_result(X_train, X_test, y_train, y_test, y_pred, 0.5, reg)


def grid_search(X, y, y_pred, y_test):
    min = 100.0
    sel_t = 0
    for i in range(30, 100, 1):
        result = valid_result(X, y, y_test, y_pred, float(i / 100))
        if result < min:
            print("update min, cur is %f" % (min))
            min = result
            sel_t = i
    print("best threshold is %f, f1_sum is %f" % (sel_t, min))


def valid_result(X_train, X_test, y_train, y_test, y_pred, error_threshold, clf):
    # 用scikit-learn计算MSE
    # mse = metrics.mean_squared_error(y_test, y_pred)

    # 模型拟合测试集
    X_outliers = [0.1, 2.1]
    X_outliers = np.row_stack((X_outliers, [-0.1, 1.9]))
    X_outliers = np.row_stack((X_outliers, [1, 2]))
    X_outliers = np.row_stack((X_outliers, [4.4, 1.9]))
    X_outliers = np.row_stack((X_outliers, [3, 0]))
    X_outliers = np.row_stack((X_outliers, [1, 1]))
    X_outliers = np.row_stack((X_outliers, [2.1, 3.2]))

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    print(str(X_outliers), str(y_pred_outliers))

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)  # 绘制异常样本的区域
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')  # 绘制正常样本和异常样本的边界
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')  # 绘制正常样本的区域
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new abnormal observations"],
               loc="upper left",
               prop=mpl.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()


def show_scatter(x_train, x_test, y_train, y_test, y_pred, reg):
    # 区域预测
    x1_min, x1_max = x_train[:, 0].min(), x_train[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x_train[:, 1].min(), x_train[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点行列均为200点
    area_smaple_point = np.stack((x1.flat, x2.flat), axis=1)  # 将区域划分为一系列测试点去用学习的模型预测，进而根据预测结果画区域
    area1_predict = reg.predict(area_smaple_point)  # 所有区域点进行预测
    area1_predict = area1_predict.reshape(
        x1.shape)  # 转化为和x1一样的数组因为用plt.pcolormesh(x1, x2, area_flag, cmap=classifier_area_color)
    # 时x1和x2组成的是200*200矩阵，area_flag要与它对应

    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 用来正常显示中文标签
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    classifier_area_color = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])  # 区域颜色
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])  # 样本所属类别颜色

    # 绘图
    # 第一个子图
    plt.figure(figsize=(16, 10))

    plt.pcolormesh(x1, x2, area1_predict, cmap=classifier_area_color)
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', s=10, cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=10, cmap=cm_dark)

    plt.xlabel('data_x', fontsize=8)
    plt.ylabel('data_y', fontsize=8)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'DecisionTreeClassifier:传统决策树', fontsize=8)
    plt.text(x1_max - 9, x2_max - 2, u'$o---train ; x---test$')

    plt.show()


def plot_show(y_pred, y_taged, y_test):
    # 建立一个矩阵，以真实标签和预测标签为元素
    print(confusion_matrix(y_test, y_taged, labels=range(2)))
    # # 交叉验证
    # predicted = cross_valid(X, classfiy_method, y)
    # # 画图
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    ax.set_title("MP")
    plt.show()


def cross_valid(X, linreg, y):
    predicted = cross_val_predict(linreg, X, y, cv=10)
    # 用scikit-learn计算MSE
    print("MSE:", metrics.mean_squared_error(y, predicted))
    # 用scikit-learn计算RMSE
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))
    return predicted


if __name__ == '__main__':
    # input_file = "E:\\mldata\\disknew\\csv\\hgst.csv";
    # sample_file = "E:\\mldata\\disknew\\csv\\hgst_sample.csv";
    # data_num = 1000
    #
    # if os.path.exists(sample_file):
    #     np_array = laod_data(sample_file)
    # else:
    #     np_array = sample_data(input_file, data_num)
    #     # np.savetxt(sample_file, np_array, fmt="%d", delimiter=",")
    #
    # smote_np = smote_data(np_array)

    x = np.random.randn(1000, 1)
    smote_np = x * 0.1
    for i in range(1):
        smote_np = np.column_stack((smote_np+1, smote_np+ 2))

    # X = np.r_[smote_np + 3, smote_np - 3]
    X = smote_np

    # np.random.shuffle(smote_np)
    # smote_np = smote_np[0:10000]

    print("data shape:" + str(X.shape))

    # X = smote_np[:, [16, 6]]
    # X = smote_np[:, [16, 6, 3, 20, 4]]
    # X = smote_np[:, [16,6,3,20,26]]
    y = np.zeros(1000)

    sgdr = SGDRegressor(loss="huber", penalty="l2")
    linreg = LinearRegression()
    lasso = LassoCV(normalize=True, max_iter=100000)

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=1000)

    ifc = IsolationForest(n_estimators=1000)
    osvm = svm.OneClassSVM(nu=0.5, kernel="rbf",
                           gamma=0.1)
    run_regress(X, y, ifc)
