
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

def classifier_train(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    if classifier_choice == 6:
            Y_pred = Y_pred > 0.8
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



def hyperParamaterFitter(classifier):
# Set some parameters for tuning
    probability_threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    smote_r = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    best_f05 = 0
    best_prob_threshold=0
    best_smote = 0
    for smote_t in smote_r:
        smote = SMOTE(ratio= smote_t,random_state = 42)
        # Splitting of the dataset (train=80%, test=20%)
        X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size = 0.2, random_state = 42)
        X_train, y_train = smote.fit_sample(X_train, y_train)
        classifier.fit(X_train, y_train)
        y_scores = classifier.predict_proba(X_test)[:,1]
        # Find best threshold
        for prob in probability_threshold:
            y_pred = y_scores > prob
            tn, fp, fn, tp = confusion_matrix(y_pred=y_pred,y_true=y_test).ravel()
            precision = 1.0 * tp / (tp + fp)
            recall = 1.0 * tp /(tp + fn)
            # Using F_0.5 score as a metric to choose best threshold
            f05 = (1 + 0.25)*(precision * recall)/(0.25 *precision + recall)
            if f05 > best_f05:
                best_smote = smote_t
                best_prob_threshold = prob
                best_f05 = f05
                best_confusion = (tn, fp, fn, tp)
                best_classifier = classifier
    return best_confusion,best_classifier,best_prob_threshold, best_smote


def cross_10_fold(classifier, features, labels, smote_v, threshold):
    k = KFold(n_splits=10, shuffle=True, random_state=42)
    TP_final = []
    FP_final = []
    TN_final = []
    FN_final = []
    AUC_final = []
    accuracy_final=[]
    f1_measure_final=[]
    Recall_final=[]
    precision_final=[]
    Y_tests_final = []
    conf_matrixes = []
    i = 0
    for train, test in k.split(features):

        print ('')
        print (' -------------------- Fold : ', i, ' --------------------')
        i = i+1
        features_train, features_test = features.iloc[train], features.iloc[test]
        labels_train, labels_test = labels.iloc[train], labels.iloc[test]

        smote = SMOTE(ratio=smote_v, random_state=42)
        features_oversampling, labels_oversampling = smote.fit_sample(features_train, labels_train)

        # classifier.fit(features_train, labels_train) , Comment out to evaluare for UNSMOTED case
        classifier.fit(features_oversampling, labels_oversampling)
        y_scores = classifier.predict_proba(features_test)[:, 1]
        labels_predicted = y_scores > threshold

        labels_predicted = classifier.predict(features_test)
        conf_table = confusion_matrix(labels_test, labels_predicted, labels=[1, 0])

        acc = accuracy_score(labels_predicted, labels_test) * 100
        f1 = f1_score(labels_predicted, labels_test) * 100
        rec = recall_score(labels_predicted, labels_test) * 100
        prec = precision_score(labels_predicted, labels_test) * 100
        conf_matrixes.append(conf_table)
        Y_tests.append(labels)
        Y_tests_preds.append(labels_predicted)
        Y_tests_preds_probabs.append(y_scores)
        TP = conf_table[0][0]
        FP = conf_table[1][0]
        TN = conf_table[1][1]
        FN = conf_table[0][1]

        labels_predicted_probability = classifier.predict_proba(features_test)[:, 1]
        AUC = roc_auc_score(labels_test, labels_predicted_probability)

        TP_final.append(TP)
        FP_final.append(FP)
        TN_final.append(TN)
        FN_final.append(FN)
        AUC_final.append(AUC)
        accuracy_final.append(acc)
        f1_measure_final.append(f1)
        Recall_final.append(rec)
        precision_final.append(prec)
        # # Smote Training Dataset
        # sm = SMOTE(ratio=0.02, random_state=42)  # 42 is the answer to the universe, life, and everything
        # X_train, Y_train = sm.fit_resample(X_train, Y_train)
        #
        # # TRAIN
        # clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
        # classifiers_array.append(clf)
        # conf_matrixes.append(conf_matrix)
        # Y_tests.append(Y_test)
        # Y_tests_preds.append(Y_pred)
        # Y_tests_preds_probabs.append(Y_pred_probab)
    TP_final = np.array(TP_final)
    FP_final = np.array(FP_final)
    TN_final = np.array(TN_final)
    FN_final = np.array(FN_final)
    AUC_final = np.array(AUC_final)
    accuracy_final=np.array(accuracy_final)
    f1_measure_final=np.array(f1_measure_final)
    recall_final=np.array(Recall_final)
    precision_final=np.array(precision_final)

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

    for i, each in enumerate(Y_tests):
        Y_tests_final.extend(each.tolist())
    Y_tests_preds_final = []
    for i, each in enumerate(Y_tests_preds):
        Y_tests_preds_final.extend(each.tolist())
    Y_tests_preds_probabs_final = []
    for i, each in enumerate(Y_tests_preds_probabs):
        Y_tests_preds_probabs_final.extend(each[:, 1].tolist())

    print (' - F1 score    : ', np.sum(f1_measure_final))
    print (' - Precision   : ', np.mean(precision_final))
    print (' - Recall      : ', np.mean(recall_final))
    roc_plot(Y_tests_final, Y_tests_preds_final, "ROC-CURVE")


# src = 'cleaned_data.csv'
src = 'df_cleaned.csv'
ENCODE_FLAG = 0

classifiers_array = []
conf_matrixes = []
Y_tests = []
Y_tests_preds = []
Y_tests_preds_probabs = []

dataset = pd.read_csv(src)

dataFrameSubset = dataset[['issuercountrycode',
                           'txvariantcode',
                           'convertedAmount',
                           'currencycode',
                           'shoppercountrycode',
                           'shopperinteraction',
                           'cardverificationcodesupplied',
                           'cvcresponsecode',
                           'accountcode',
                           'simple_journal']]

#ad-hoc endoding of features
dataFrameSubset.loc[dataFrameSubset.simple_journal == 'Chargeback', 'simple_journal'] = 1
dataFrameSubset.loc[dataFrameSubset.simple_journal == 'Settled', 'simple_journal'] = 0
dataFrameSubset['simple_journal'] = dataFrameSubset['simple_journal'].astype('int')
labels = dataFrameSubset.simple_journal
features = dataFrameSubset.drop('simple_journal', axis=1)
features = pd.get_dummies(features)
features.info()

##
# Instead of usx, usy use features, labels
##
# dataset['n_amount'] = StandardScaler().fit_transform(dataset['amount'].values.reshape(-1,1))
# dataset['n_conv_amount'] = StandardScaler.fit_transform(dataset['con'])

# x = dataset[:,:-1]
# y = dataset[:,-1]
#
# TP, FP, FN, TN = 0, 0, 0, 0
# x_array = np.array(x)
# y_array = np.array(y)
# usx = x_array
# usy = y_array
# usx = usx.astype(np.float64)
# usy = usy.astype(np.float64)

features_array = [0,1,2,3,5,6,7,8,9,10]

# Feature Selection - Based on Viz Earlier
# if ENCODE_FLAG:
#     encode_x = usx[:,[1,2,7,8]]
#     usx = usx[:,[0]]
#     # usx = normalize(usx)
#
#     enc = OneHotEncoder(handle_unknown='ignore').fit(encode_x)
#     encode_x = enc.transform(encode_x).toarray()
#     usx = np.concatenate((usx, encode_x), axis=1)
#
# else:
#     usx = usx[:, [0,1,2,3,4,7,8]]

#Normalize features
# usx = normalize(usx)

# print "Shape of feature selected dataset: " + str(usx.shape)
print("Training Beginssssss!!!!!!!")


#Classifier choice
classifier_choice = 5
classifiers = [
    neighbors.KNeighborsClassifier(n_neighbors=5),          # ----> 1
    SVC(),                                                  # ----> 2
    DecisionTreeClassifier(),                               # ----> 3
    RandomForestClassifier(),                               # ----> 4
    AdaBoostClassifier(),                                   # ----> 5
    LogisticRegression(C=400, penalty='l1'),                # ----> 6
    # MLPClassifier(solver='adam', alpha=1e-2)                # ----> 7

    ]

clf = classifiers[classifier_choice-1]

# (tn, fp, fn, tp),best_classifier, BestThreshold, best_smote = hyperParamaterFitter(clf)
# print ((tn, fp, fn, tp))
# print best_classifier
# print BestThreshold
# print best_smote
#
# # x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)#test_size: proportion of train/test data
# for i, (train, test) in enumerate(StratifiedKFold(n_splits=10, random_state=25).split(usx, usy)):
#     print ('')
#     print (' -------------------- Fold : ', i, ' --------------------')
#
#     X_train = usx[train]
#     Y_train = usy[train]
#     X_test  = usx[test]
#     Y_test  = usy[test]
#
#     X_train, Y_train = shuffle(X_train, Y_train)
#
#     # Under Sampling - Tomek removal
#     tl = TomekLinks()
#     X_train, Y_train = tl.fit_resample(X_train, Y_train)
#     # Smote Training Dataset
#     sm = SMOTE(ratio=0.02, random_state=42) # 42 is the answer to the universe, life, and everything
#     X_train, Y_train = sm.fit_resample(X_train, Y_train)
#     print (' -------------------- Done with Under/Over Sampling --------------------')
#
#
#     # TRAIN
#     clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
#     classifiers_array.append(clf)
#     conf_matrixes.append(conf_matrix)
#     Y_tests.append(Y_test)
#     Y_tests_preds.append(Y_pred)
#     Y_tests_preds_probabs.append(Y_pred_probab)
#
# # Confusion Matrices
# conf_matrix_final = []
# for i, each in enumerate(conf_matrixes):
#     if i == 0:
#         conf_matrix_final = each.copy()
#     else:
#         conf_matrix_final += each.copy()
#
# print '\n'
# print '\n'
# print (' - - - - - - Final Conf Matrix - - - - - - ')
# print (conf_matrix_final)
#
# # ROC-CURVEs
# Y_tests_final = []
# for i, each in enumerate(Y_tests):
#     Y_tests_final.extend(each.tolist())
# Y_tests_preds_final = []
# for i, each in enumerate(Y_tests_preds):
#     Y_tests_preds_final.extend(each.tolist())
# Y_tests_preds_probabs_final = []
# for i, each in enumerate(Y_tests_preds_probabs):
#     Y_tests_preds_probabs_final.extend(each[:, 1].tolist())
#
# print (' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# print (' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# print (' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# roc_plot(Y_tests_final, Y_tests_preds_final, "FIX TITLE")

# (tn, fp, fn, tp),best_classifier, BestThreshold, best_smote = hyperParamaterFitter(clf)
# print ((tn, fp, fn, tp))
# print best_classifier
# print BestThreshold
# print best_smote

cross_10_fold(clf,features,labels,0.2,0.6)

# # x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)#test_size: proportion of train/test data
# for i, (train, test) in enumerate(StratifiedKFold(n_splits=10, random_state=25).split(usx, usy)):
#     print ('')
#     print (' -------------------- Fold : ', i, ' --------------------')
#
#     X_train = usx[train]
#     Y_train = usy[train]
#     X_test  = usx[test]
#     Y_test  = usy[test]
#
#     X_train, Y_train = shuffle(X_train, Y_train)
#
#     # Smote Training Dataset
#     sm = SMOTE(ratio=0.02, random_state=42) # 42 is the answer to the universe, life, and everything
#     X_train, Y_train = sm.fit_resample(X_train, Y_train)
#
#     # TRAIN
#     clf, conf_matrix, Y_pred_probab, Y_pred = classifier_train(clf, X_train, Y_train, X_test, Y_test)
#     classifiers_array.append(clf)
#     conf_matrixes.append(conf_matrix)
#     Y_tests.append(Y_test)
#     Y_tests_preds.append(Y_pred)
#     Y_tests_preds_probabs.append(Y_pred_probab)
#
# # Confusion Matrices
# conf_matrix_final = []
# for i, each in enumerate(conf_matrixes):
#     if i == 0:
#         conf_matrix_final = each.copy()
#     else:
#         conf_matrix_final += each.copy()
#
# print '\n'
# print '\n'
# print (' - - - - - - Final Conf Matrix - - - - - - ')
# print (conf_matrix_final)
#
# # ROC-CURVEs
# Y_tests_final = []
# for i, each in enumerate(Y_tests):
#     Y_tests_final.extend(each.tolist())
# Y_tests_preds_final = []
# for i, each in enumerate(Y_tests_preds):
#     Y_tests_preds_final.extend(each.tolist())
# Y_tests_preds_probabs_final = []
# for i, each in enumerate(Y_tests_preds_probabs):
#     Y_tests_preds_probabs_final.extend(each[:, 1].tolist())
#
# print (' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# print (' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# print (' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
# roc_plot(Y_tests_final, Y_tests_preds_final, "FIX TITLE")

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

