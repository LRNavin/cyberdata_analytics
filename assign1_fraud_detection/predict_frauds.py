
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE


def classifier_train(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    Y_pred_probab = clf.predict_proba(X_test)
    print (' - Conf Matrix : ')
    print (conf_matrix)
    print (' - F1 score    : ', round(metrics.f1_score(Y_test, Y_pred, pos_label=1), 3))
    print (' - Precision   : ', round(metrics.precision_score(Y_test, Y_pred, pos_label=1), 3))
    print (' - Recall      : ', round(metrics.recall_score(Y_test, Y_pred, pos_label=1), 3))

    return clf, conf_matrix, Y_pred_probab, Y_pred


def roc_plot(Y_test, Y_pred, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


src = 'cleaned_data.csv'
ENCODE_FLAG = 0

classifiers_array = []
conf_matrixes = []
Y_tests = []
Y_tests_preds = []
Y_tests_preds_probabs = []

dataset = pd.read_csv(src).as_matrix()

x = dataset[:,:-1]
y = dataset[:,-1]

TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)
y_array = np.array(y)
usx = x_array
usy = y_array
usx = usx.astype(np.float64)
usy = usy.astype(np.float64)

# Feature Selection - Based on Viz Earlier
if ENCODE_FLAG:
    encode_x = usx[:,[1,2,7,8]]
    usx = usx[:,[0]]
    # usx = normalize(usx)

    enc = OneHotEncoder(handle_unknown='ignore').fit(encode_x)
    encode_x = enc.transform(encode_x).toarray()
    usx = np.concatenate((usx, encode_x), axis=1)

else:
    usx = usx[:, [0,1,2,7,8]]


#Normalize features
# usx = normalize(usx)

print "Shape of feature selected dataset: " + str(usx.shape)
print("Training Beginssssss!!!!!!!")


#Classifier choice
classifier_choice = 6
classifiers = [
    neighbors.KNeighborsClassifier(algorithm='kd_tree'),    # ----> 1
    SVC(),                                                  # ----> 2
    DecisionTreeClassifier(),                               # ----> 3
    RandomForestClassifier(),                               # ----> 4
    AdaBoostClassifier(),                                   # ----> 5
    LogisticRegression(C=400, penalty='l1')                 # ----> 6
    ]

clf = classifiers[classifier_choice-1]

# x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)#test_size: proportion of train/test data

for i, (train, test) in enumerate(StratifiedKFold(n_splits=10, random_state=25).split(usx, usy)):
    print ('')
    print (' -------------------- Fold : ', i, ' --------------------')

    X_train = usx[train]
    Y_train = usy[train]
    X_test  = usx[test]
    Y_test  = usy[test]

    X_train, Y_train = shuffle(X_train, Y_train)

    # Smote Training Dataset
    sm = SMOTE(ratio=0.01)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)

    # TRAIN
    clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
    classifiers_array.append(clf)
    conf_matrixes.append(conf_matrix)
    Y_tests.append(Y_test)
    Y_tests_preds.append(Y_pred)
    Y_tests_preds_probabs.append(Y_pred_probab)

# Confusion Matrices
conf_matrix_final = []
for i, each in enumerate(conf_matrixes):
    if i == 0:
        conf_matrix_final = each.copy()
    else:
        conf_matrix_final += each.copy()

print '\n'
print '\n'
print (' - - - - - - Final Conf Matrix - - - - - - ')
print (conf_matrix_final)

# ROC-CURVEs
Y_tests_final = []
for i, each in enumerate(Y_tests):
    Y_tests_final.extend(each.tolist())
Y_tests_preds_final = []
for i, each in enumerate(Y_tests_preds):
    Y_tests_preds_final.extend(each.tolist())
Y_tests_preds_probabs_final = []
for i, each in enumerate(Y_tests_preds_probabs):
    Y_tests_preds_probabs_final.extend(each[:, 1].tolist())

print (' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
print (' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
print (' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
roc_plot(Y_tests_final, Y_tests_preds_final, "FIX TITLE")

# y_predict = cross_val_predict(clf, x_train, y_train, cv=10)

# y_cross_validate = cross_validate(clf, x_train, y_train, cv=10, return_train_score=True)

# y_predict = clf.predict(x_test)

# for i in range(len(y_predict)):
#     if y_train[i]==1 and y_predict[i]==1:
#         TP += 1
#     if y_train[i]==0 and y_predict[i]==1:
#         FP += 1
#     if y_train[i]==1 and y_predict[i]==0:
#         FN += 1
#     if y_train[i]==0 and y_predict[i]==0:
#         TN += 1
# print('TP: '+ str(TP))
# print('FP: '+ str(FP))
# print('FN: '+ str(FN))
# print('TN: '+ str(TN))

# print "------------------------Confusion Matrix------------------------"
# print confusion_matrix(y_true=y_train, y_pred=y_predict) #watch out the element in confusion matrix
#
#
# precision, recall, thresholds = precision_recall_curve(y_train, y_predict)
#
# print('Accuracy: ' + str(accuracy_score(y_true=y_train, y_pred=y_predict)))
# print('Precision: '+ str(precision))
# print('Recall: '+ str(recall))
#
# fpr, tpr, thresholds = metrics.roc_curve(y_train, y_predict)
#
# # calculate AUC
# auc = metrics.roc_auc_score(y_train, y_predict)
# print('AUC: %.3f' % auc)
#
# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr, tpr, marker='.')
# # show the plot
# plt.show()

#predict_proba = clf.predict_proba(x_test)#the probability of each smple labelled to positive or negative

