
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

src = 'cleaned_data.csv'
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

usx = usx[:,[0,8]]

print "Shape of feature selected dataset: " + str(usx.shape)

x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2)#test_size: proportion of train/test data

print("Training Beginssssss!!!!!!!")
print("Trainging Size: " + str(x_train.shape[0]))
print("Testing Size: " + str(x_test.shape[0]))
print ("Testing Frauds: " + str(np.sum(y_test)))

#Classifier choice
classifier_choice = 4
classifiers = [
    neighbors.KNeighborsClassifier(algorithm='kd_tree'),    # ----> 1
    SVC(),                                                  # ----> 2
    DecisionTreeClassifier(max_depth=5),                    # ----> 3
    RandomForestClassifier(max_depth=5, n_estimators=10)    # ----> 4
    ]

clf = classifiers[classifier_choice-1]
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
for i in range(len(y_predict)):
    if y_test[i]==1 and y_predict[i]==1:
        TP += 1
    if y_test[i]==0 and y_predict[i]==1:
        FP += 1
    if y_test[i]==1 and y_predict[i]==0:
        FN += 1
    if y_test[i]==0 and y_predict[i]==0:
        TN += 1
print('TP: '+ str(TP))
print('FP: '+ str(FP))
print('FN: '+ str(FN))
print('TN: '+ str(TN))

print "------------------------Confusion Matrix------------------------"
print confusion_matrix(y_true=y_test, y_pred=y_predict) #watch out the element in confusion matrix


precision, recall, thresholds = precision_recall_curve(y_test, y_predict)

print('Accuracy: ' + str(accuracy_score(y_true=y_test, y_pred=y_predict)))
print('Precision: '+ str(precision))
print('Recall: '+ str(recall))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)

# calculate AUC
auc = metrics.roc_auc_score(y_test, y_predict)
print('AUC: %.3f' % auc)

# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

#predict_proba = clf.predict_proba(x_test)#the probability of each smple labelled to positive or negative

