print("r_SiN_change")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,auc,accuracy_score,recall_score,precision_score

data_input = pd.read_csv(r"E:\zhaoxue(last-first)_2\activity_plus\3.csv")

X_new = pd.concat([

    # babblespeech
      data_input.iloc[:, data_input.columns == "bs_ch11_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch12_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch13_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch14_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch15_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch16_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch17_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch18_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch19_change"]
    , data_input.iloc[:, data_input.columns == "bs_ch20_change"]



], axis=1)
Y_new = data_input.loc[:, "score"]
data_value = pd.concat([Y_new, X_new], axis=1)
X = data_value.iloc[:, data_value.columns != "score"]
Y = data_value.iloc[:, data_value.columns == "score"]
Y = np.array(Y)
columns = X.columns
X = StandardScaler().fit_transform(X)  # 标准化

# analysis
import warnings

warnings.filterwarnings("ignore")

svc = SVC(kernel="linear", probability=True)
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)}

accuracy_list = []
precision_list = []
recall_list = []
auc_list = []

for i in range(1, 101):  # 100次嵌套交叉验证
    print(i)
    accuracy = 0
    precision = 0
    recall = 0
    area = 0
    for o_train, o_test in StratifiedKFold(n_splits=10, shuffle=True, random_state=i).split(X, Y):

        # 特征选择
        compare_score = 0
        compare_index = 0
        for s in range(1, 11, 1):  # 在训练集X[O_train]上做交叉验证，选出最优特征子集
            X_wrapper = RFE(svc, n_features_to_select=s, step=1).fit_transform(X[o_train], Y[o_train])
            once = cross_val_score(svc, X_wrapper, Y[o_train], scoring='accuracy'
                                   , cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=i)).mean()
            # once记录选择s个特征时，在训练集上进行一次交叉验证的结果
            if once > compare_score:  # 记录准确率最高的特征数量
                compare_score = once
                compare_index = s
        # print('selected feature number:%d' % compare_index)#循环结束，打印最优特征子集的特征个数

        # 在进行网格搜索前，将X[o_train],X[o_test]的特征数量转化为compare_index个
        rfe_select_wrapper = RFE(svc, n_features_to_select=compare_index, step=1).fit(X[o_train], Y[o_train])
        Xtrain_wrapper_f = rfe_select_wrapper.transform(X[o_train])  # 收缩X[o_train]上的特征
        Xtest_wrapper_f = rfe_select_wrapper.transform(X[o_test])  # 说说X[o_test]上的特征

        # 在Xtrain_wrapper_f上进行网格搜索（基于选出特征的X[o_train]上进行网格搜索）
        grid = GridSearchCV(svc, parameters, scoring='accuracy',
                            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=i))
        grid = grid.fit(Xtrain_wrapper_f, Y[o_train])
        best_estimator = grid.best_estimator_  # 最好的模型

        # model evaluation
        acc = accuracy_score(Y[o_test], best_estimator.predict(Xtest_wrapper_f))
        accuracy += acc

        prec = precision_score(Y[o_test], best_estimator.predict(Xtest_wrapper_f))
        precision += prec

        rec = recall_score(Y[o_test], best_estimator.predict(Xtest_wrapper_f))
        recall += rec

        probas_ = best_estimator.predict_proba(Xtest_wrapper_f)
        fpr, tpr, thresholds = roc_curve(Y[o_test], probas_[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        area += roc_auc

    accuracy_list.append(accuracy / 10)
    precision_list.append(precision / 10)
    recall_list.append(recall / 10)
    auc_list.append(area / 10)

pd.DataFrame(accuracy_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\classification\\r_SiN_change(acc100).csv')
pd.DataFrame(precision_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\classification\\r_SiN_change(pre100).csv')
pd.DataFrame(recall_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\classification\\r_SiN_change(rec100).csv')
pd.DataFrame(auc_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\classification\\r_SiN_change(auc100).csv')