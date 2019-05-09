#Import Libraries
import os
import pandas as pd
import requests
import json
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

#Remove refused transactions from dataset
cwd = os.getcwd()
dataFrame = pd.read_csv(cwd+"/adyen_data.csv")
dataFrame = dataFrame.loc[dataFrame['simple_journal'] != 'Refused']
dataFrame.info()
#convert booking date and creation date to DataTime
dates=['bookingdate', 'creationdate']
for col in dates:
    dataFrame[col] = pd.to_datetime(dataFrame.bookingdate, format='%Y-%m-%d %H:%M:%S', errors='coerce')
#Handle the following attributes as categorical
categorical=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
               'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']
for col in categorical:
    dataFrame[col] = dataFrame[col].astype('category')

url = 'https://api.exchangerate-api.com/v4/latest/EUR'
response = requests.get(url)
data = json.loads(response.text)
rates = data['rates']

coversion_dict = {key: float(rates[key]) for key in dataFrame.currencycode.unique()}
# Calculate converted amount of money
def convert_currency(row):
  currency = row['currencycode']
  amount = row['amount']
  return round(amount * coversion_dict[currency] / 100)

# Clean - amount
dataFrame['convertedAmount'] = dataFrame.apply(lambda x: convert_currency(x),axis=1)
dataFrame['convertedAmount'] = StandardScaler().fit_transform(dataFrame['amount'].values.reshape(-1,1))

##Preprocessing stage finished