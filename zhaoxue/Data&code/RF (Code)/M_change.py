
import pandas as pd

data_input = pd.read_csv(r"E:\zhaoxue(last-first)_2\activity_plus\3.csv")
# print("读入的非空数据:\n",data_input)

X_new = pd.concat([


      data_input.iloc[:, data_input.columns == "m_ch1_change"]
    , data_input.iloc[:, data_input.columns == "m_ch2_change"]
    , data_input.iloc[:, data_input.columns == "m_ch3_change"]
    , data_input.iloc[:, data_input.columns == "m_ch4_change"]
    , data_input.iloc[:, data_input.columns == "m_ch5_change"]
    , data_input.iloc[:, data_input.columns == "m_ch6_change"]
    , data_input.iloc[:, data_input.columns == "m_ch7_change"]
    , data_input.iloc[:, data_input.columns == "m_ch8_change"]
    , data_input.iloc[:, data_input.columns == "m_ch9_change"]
    , data_input.iloc[:, data_input.columns == "m_ch10_change"]


    , data_input.iloc[:, data_input.columns == "m_ch11_change"]
    , data_input.iloc[:, data_input.columns == "m_ch12_change"]
    , data_input.iloc[:, data_input.columns == "m_ch13_change"]
    , data_input.iloc[:, data_input.columns == "m_ch14_change"]
    , data_input.iloc[:, data_input.columns == "m_ch15_change"]
    , data_input.iloc[:, data_input.columns == "m_ch16_change"]
    , data_input.iloc[:, data_input.columns == "m_ch17_change"]
    , data_input.iloc[:, data_input.columns == "m_ch18_change"]
    , data_input.iloc[:, data_input.columns == "m_ch19_change"]
    , data_input.iloc[:, data_input.columns == "m_ch20_change"]

], axis=1)
Y_new = data_input.loc[:, "score"]

data_value = pd.concat([Y_new, X_new], axis=1)
# print(data_value.info())

# --------------------------------------------------------
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn import svm
from sklearn.metrics import roc_curve, auc , accuracy_score ,recall_score,precision_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut,KFold
from sklearn.metrics import mean_squared_error as MSE
# --------------------------------------------------------


X = data_value.iloc[:, data_value.columns != "score"]
Y = data_value.iloc[:, data_value.columns == "score"]
columns = X.columns
Y = np.array(Y)
#X = np.array(X)

X = StandardScaler().fit_transform(X)#标准化
#Y = StandardScaler().fit_transform(Y)


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

RFC_ = RFC(n_estimators=10, random_state=0)
acc_list = []
pre_list = []
rec_list = []
auc_list= []
for i in range(1, 101):
    accuracy = 0
    precision = 0
    recall = 0
    area=0
    for train, test in StratifiedKFold(n_splits=10, shuffle=True, random_state=i).split(X, Y):
        Xtrain_column = columns
        threshold_ = np.linspace(0, (RFC_.fit(X[train], Y[train]).feature_importances_).max(), 10)
        #print(test)
        compare_score = 0
        compare_index = 0
        for s in threshold_:
            X_embedded = SelectFromModel(RFC_, threshold=s).fit_transform(X[train], Y[train])
            once = cross_val_score(RFC_, X_embedded, Y[train], cv=10).mean()
            if once > compare_score:  # 记录准确率最高的特征数量
                compare_score = once
                compare_index = s

        sfm_select_wrapper = SelectFromModel(RFC_, threshold=compare_index).fit(X[train], Y[train])
        Xtrain_wrapper_f = sfm_select_wrapper.transform(X[train])  # 收缩X[train]上的特征
        Xtest_wrapper_f = sfm_select_wrapper.transform(X[test])  # 收缩X[test]上的特征

        RFC__ = RFC_.fit(Xtrain_wrapper_f, Y[train])
        # print(sfm_select_wrapper.get_support())
        # print([*zip(Xtrain_column[sfm_select_wrapper.get_support()],RFC__.feature_importances_)])
        pd.DataFrame([*zip(Xtrain_column[sfm_select_wrapper.get_support()]
                    , RFC__.feature_importances_)]).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\RF\\contributing_weight(M_change).csv'
                                                           ,mode='a')
        # print(pd.DataFrame(Xtest_wrapper_f))

        acc = accuracy_score(Y[test], RFC__.predict(Xtest_wrapper_f))
        accuracy += acc

        pre=precision_score(Y[test], RFC__.predict(Xtest_wrapper_f))
        precision += pre

        rec=recall_score(Y[test], RFC__.predict(Xtest_wrapper_f))
        recall += rec

        probas_ = RFC__.predict_proba(Xtest_wrapper_f)
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        area += roc_auc

    acc_list.append(accuracy / 10)
    pre_list.append(precision / 10)
    rec_list.append(recall / 10)
    auc_list.append(area / 10)


pd.DataFrame(acc_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\RF\\acc(M_change).csv')
pd.DataFrame(pre_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\RF\\pre(M_change).csv')
pd.DataFrame(rec_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\RF\\rec(M_change).csv')
pd.DataFrame(auc_list).to_csv('E:\\zhaoxue(last-first)_2\\activity_plus\\RF\\auc(M_change).csv')