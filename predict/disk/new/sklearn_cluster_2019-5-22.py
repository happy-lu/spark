# coding=utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from data_normalize import *
from data_sampling import *
from sklearn.metrics import auc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


def run_and_show_detail(X, y, name, reg):
    print("use reg:" + str(reg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    if str(type(reg)) == "<class 'sklearn.svm.classes.OneClassSVM'>":
        print("OneClassSVM, need remove failure==1 data")
        filter = y_train == 0
        X_train = X_train[filter]
        y_train = y_train[filter]
        print("OneClassSVM, new data shape is", X_train.shape, y_train.shape)

    reg.fit(X_train, y_train)

    if hasattr(reg, "intercept_"):
        print(reg.intercept_)
    if hasattr(reg, "coef_"):
        print(reg.coef_)
    if hasattr(reg, "feature_importances_"):
        if X.shape[1] == 5:
            names = ["smart_9_raw", "smart_2_raw", "smart_196_raw", "smart_12_raw", "smart_4_raw"]
        else:
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
    print (y_pred)

    # if hasattr(reg, "score"):
    #     print("predict accuracy：", reg.score(X_test, y_test))
    #
    # return valid_and_show_detail(X_train, X_test, y_train, y_test, y_pred, name, reg)


def valid_and_show_detail(X_train, X_test, y_train, y_test, y_pred, name, reg, show_pr_plot=False):
    if show_pr_plot:
        precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_pred)
        fig, ax = plt.subplots()
        plt.plot(precision, recall, marker='o')
        ax.set_title("PR")
        plt.show()

    # show_scatter(X_train, X_test, y_train, y_test, y_pred, reg)
    return cal_result(y_test, y_pred, name, reg)

    # # 将结果划分成0，1
    # y_taged = copy.deepcopy(y_pred)
    # y_taged[y_taged > error_threshold] = 1.0
    # y_taged[y_taged <= error_threshold] = 0.0
    # # 通过该函数，比较预测出的标签和真实标签，并输出准确率
    #
    # result_json = classification_report(y_test, y_taged, output_dict=True)
    # f1_sum = result_json["1.0"]["f1-score"]
    #
    # print("threshold: " + str(error_threshold) + ",f1_sum: " + str(f1_sum))
    # print(str(result_json))
    #
    # plot_show(y_pred, y_taged, y_test)
    #
    # return f1_sum


def cal_result(y_test, y_pred, name, reg):
    if name == "ocs" or name == "if":
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        print("failed==1's origin value:", y_test[y_pred == 1])
    elif name == "linreg" or name == "lasso":
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        print("failed==1's origin value:", y_test[y_pred == 1])

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse)
    print("RMSE:", rmse)

    r2 = r2_score(y_test, y_pred)
    print("r2:", r2)
    print("others:\n", classification_report(y_test, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print("fpr:", fpr, "tpr:", tpr, "thresholds:", thresholds)

    AUC = auc(fpr, tpr)
    print("AUC:", AUC)

    return AUC, fpr, tpr, r2, mse, rmse


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


def cross_valid(X, y, name, reg):
    y_pred = cross_val_predict(reg, X, y, cv=10)

    return cal_result(y, y_pred, name, reg)


def lstm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # 设置最大特征的数量，对于文本，就是处理的最大单词数量。若被设置为整数，则被限制为待处理数据集中最常见的max_features个单词
    max_features = 20000
    # 设置训练的轮次
    batch_size = 32

    # 创建网络结构
    model = Sequential()
    model.add(Embedding(len(X_train), 2, input_length=5))
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
    return val_mean


def sklearn_method(np_array):
    np_smote = smote_enn_data(np_array)
    avl_methods = {"SGDClassifier": SGDClassifier(loss="huber", penalty="l2"),
                   "LinearRegression": LinearRegression(),
                   "LassoCV": LassoCV(normalize=True, max_iter=100000),
                   "DecisionTreeClassifier": DecisionTreeClassifier(),
                   "RandomForestClassifier": RandomForestClassifier(n_estimators=1000),
                   "AdaBoostClassifier": AdaBoostClassifier(
                       DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                       algorithm="SAMME",
                       n_estimators=1000, learning_rate=0.5),
                   "IsolationForest": IsolationForest(n_estimators=1000),
                   "OneClassSVM": svm.OneClassSVM(nu=0.5, kernel="poly",
                                                  gamma=0.007)
                   }
    avl_methods = {
        "RandomForestClassifier": RandomForestRegressor(n_estimators=1000)}
    maxAuc = 0
    result_dict = {}
    for smote in range(2):
        data = np_array if smote == 0 else np_smote

        X = data[:, 1:]
        # X = data[:, [16, 6]]
        X = data[:, [16, 4, 24, 20, 8]]
        # X = data[:, [16,6,3,20,26]]
        y = data[:, 0]
        for stand in range(2):
            cal_X = X if stand == 0 else standard(X)
            cal_y = y

            for name, reg in avl_methods.items():
                print("use method:%s, use_smote: %s, use_stand: %s" % (name, smote, stand))
                # AUC, fpr, tpr, r2, mse, rmse = cross_valid(cal_X, cal_y, name, reg)
                run_and_show_detail(X, y, name, reg)
                # AUC, fpr, tpr, r2, mse, rmse = run_and_show_detail(X, y, name, reg)
                #
                # result_dict[name, smote, stand] = AUC
                # if AUC > maxAuc:
                #     maxAuc = AUC
                #     best_method = name
                #     use_smote = smote
                #     use_stand = stand
    # print(
    #     "best_method: %s, use_smote: %s, use_stand: %s, maxAuc is: %s" % (best_method, use_smote, use_stand, maxAuc))
    # result_dict = sorted(result_dict.items(), key=lambda d: d[1])
    # print("result map is:\n", result_dict)


if __name__ == '__main__':
    input_file = "E:\\mldata\\disknew\\csv\\8t_all.csv";
    sample_file = "E:\\mldata\\disknew\\csv\\8t_all_sample.csv";
    data_num = 1000

    if os.path.exists(sample_file):
        np_array = laod_data(sample_file)
    else:
        np_array = sample_data(input_file, data_num)
        np.savetxt(sample_file, np_array, fmt="%d", delimiter=",")

    # np.random.shuffle(smote_np)
    # smote_np = smote_np[0:10000]

    print("data shape:" + str(np_array.shape))
    sklearn_method(np_array)

    # np_smote = smote_enn_data(np_array)
    # X = np_smote[:, [16, 4, 20, 24, 28]]
    # # X = data[:, [16,6,3,20,26]]
    # X = min_max(X)
    # y = np_smote[:, 0]
    #
    # lstm(X, y)
