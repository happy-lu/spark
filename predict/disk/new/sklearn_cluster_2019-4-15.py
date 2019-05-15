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

import copy
import json
import matplotlib.pyplot as plt


def svm(X, y):
    pass


def kemans(X, y):
    pass


def isolation_forest(X, y):
    pass


def decision_tree(X, y):
    pass


def logic_regress(X, y):
    pass


def run_regress(X, y, reg):
    print("use reg:" + str(reg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    reg.fit(X_train, y_train)


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
    y_pred = reg.predict(X_test)

    print("predict accuracy：", reg.score(X_test, y_test))

    # grid_search(X, y, y_pred, y_test)
    valid_result(X, y, y_test, y_pred, 0.5)


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


def valid_result(X, y, y_test, y_pred, error_threshold, classfiy_method=None):
    # 用scikit-learn计算MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse)
    # 用scikit-learn计算RMSE
    print("RMSE:", rmse)
    r2 = r2_score(y_test, y_pred)
    print("r2:", r2)
    print("others:\n", classification_report(y_test, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print("fpr:", fpr, "tpr:", tpr, "thresholds:", thresholds)


    from sklearn.metrics import auc
    AUC = auc(fpr, tpr)
    print("AUC:", AUC)

    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_pred)

    fig, ax = plt.subplots()
    plt.plot(precision, recall, marker='o')
    ax.set_title("PR")
    plt.show()

    # 将结果划分成0，1
    y_taged = copy.deepcopy(y_pred)
    y_taged[y_taged > error_threshold] = 1.0
    y_taged[y_taged <= error_threshold] = 0.0
    # 通过该函数，比较预测出的标签和真实标签，并输出准确率

    result_json = classification_report(y_test, y_taged, output_dict=True)
    f1_sum = result_json["1.0"]["f1-score"]

    print("threshold: " + str(error_threshold) + ",f1_sum: " + str(f1_sum))
    print(str(result_json))

    plot_show(y_pred, y_taged, y_test)

    return f1_sum


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
    input_file = "E:\\mldata\\disknew\\csv\\hgst.csv";
    sample_file = "E:\\mldata\\disknew\\csv\\hgst_sample.csv";
    data_num = 1000

    if os.path.exists(sample_file):
        np_array = laod_data(sample_file)
    else:
        np_array = sample_data(input_file, data_num)
        # np.savetxt(sample_file, np_array, fmt="%d", delimiter=",")

    smote_np = smote_data(np_array)
    print("data shape:" + str(smote_np.shape))

    X = smote_np[:, 1:]
    y = smote_np[:, 0]

    X = standard(X)

    sgdr = SGDRegressor(loss="huber", penalty="l2")
    linreg = LinearRegression()
    lasso = LassoCV(normalize=True, max_iter=100000)

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=10)

    run_regress(X, y, dt)
