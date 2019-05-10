
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression, LinearRegression

from imblearn.over_sampling import SMOTE

classifiers_array = []
conf_matrixes = []
Y_tests = []
Y_tests_preds = []
Y_tests_preds_probabs = []

classifiers = [
    LogisticRegression(C=400, penalty='l1'),  # ----> 1
    AdaBoostClassifier()                      # ----> 2

]

classifiers_name = [
    "Logistic Regression",
    "AdaBoost"
]


class FraudData_Classificaiton():


    def roc_plot(self, Y_test, Y_pred, title=''):
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

    def cross_10_fold(self, classifier, features, labels, classifier_choice):

        conf_matrixes = []

        print("Training Beginssssss!!!!!!!")

        for i, ( train, test ) in enumerate(StratifiedKFold(n_splits=5, shuffle=True).split(features, labels)):

            print ('')
            print (' -------------------- Fold : ', i, ' -------------------- ')

            features_train, features_test = features.iloc[train], features.iloc[test]
            labels_train, labels_test = labels.iloc[train], labels.iloc[test]
            print "Features Located"

            smote = SMOTE(ratio=0.5)
            features_oversampling, labels_oversampling = smote.fit_resample(features_train, labels_train)

            print "Smote Done"

            classifier.fit(features_oversampling, labels_oversampling)
            y_scores = classifier.predict_proba(features_test)[:, 1]
            labels_predicted = classifier.predict(features_test)

            conf_table = confusion_matrix(labels_test, labels_predicted)

            print "Fold " + str(i) + " Conf.Matrix"
            print conf_table

            conf_matrixes.append(conf_table)
            Y_tests.append(labels_test)
            Y_tests_preds.append(labels_predicted)
            Y_tests_preds_probabs.append(y_scores)


        # Confusion Matrices
        conf_matrix_final = []
        for i, each in enumerate(conf_matrixes):
            if i == 0:
                conf_matrix_final = each.copy()
            else:
                conf_matrix_final += each.copy()

        print '\n'
        print (' - - - - - - Final Conf Matrix - - - - - - ')
        print (conf_matrix_final)

        Y_tests_final = []
        for i, each in enumerate(Y_tests):
            Y_tests_final.extend(each.tolist())

        Y_tests_preds_final = []
        for i, each in enumerate(Y_tests_preds):
            Y_tests_preds_final.extend(each.tolist())

        Y_tests_preds_proba_final = []
        for i, each in enumerate(Y_tests_preds_probabs):
            Y_tests_preds_proba_final.extend(each.tolist())

        print (' - F1 score    : ', round(metrics.f1_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
        print (' - Precision   : ', round(metrics.precision_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))
        print (' - Recall      : ', round(metrics.recall_score(Y_tests_final, Y_tests_preds_final, pos_label=1), 3))

        self.roc_plot(Y_tests_final, Y_tests_preds_proba_final, "ROC curve - " + str(classifiers_name[classifier_choice-1]))


    def trigger_run(self, classifier_choice):

        src = 'df_cleaned.csv'
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
        dataFrameSubset.loc[dataFrameSubset.simple_journal == 'Chargeback', 'simple_journal'] = 0
        dataFrameSubset.loc[dataFrameSubset.simple_journal == 'Settled', 'simple_journal'] = 1
        dataFrameSubset['simple_journal'] = dataFrameSubset['simple_journal'].astype('int')
        labels = dataFrameSubset.simple_journal
        features = dataFrameSubset.drop('simple_journal', axis=1)
        features = pd.get_dummies(features)
        features.info()

        features_array = features.as_matrix()
        print "Shape of the training features: " + str(features_array.shape)


        clf = classifiers[classifier_choice-1]
        self.cross_10_fold(clf, features, labels, classifier_choice)
