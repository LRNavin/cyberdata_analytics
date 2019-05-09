# import packages
# from google.colab import drive

import os
import seaborn as sns
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')


#Data Explore
# src = 'data_for_student_case.csv'
# drive.mount('/content/drive')

src = 'adyen_data.csv'
df = pd.read_csv(src)

# df_clean = df.copy()
df = df[df['simple_journal']!='Refused']

df['creationdate'] = pd.to_datetime(df['creationdate'],format='%Y-%m-%d %H:%M:%S')
# df['creation_date'] = pd.to_datetime(df['creationdate'],format='%Y-%m-%d %H:%M:%S')
# df['creation_month'] = df.creation_date.dt.month
# df['creation_weekday'] = df.creation_date.dt.weekday
# df['creation_day'] = df.creation_date.dt.date

url = 'https://api.exchangerate-api.com/v4/latest/EUR'
response = requests.get(url)
data = json.loads(response.text)
rates = data['rates']

coversion_dict = {key: float(rates[key]) for key in df.currencycode.unique()}
# Calculate converted amount of money
def convert_currency(row):
  currency = row['currencycode']
  amount = row['amount']
  return round(amount * coversion_dict[currency] / 100)

dataset = pd.DataFrame()

# Clean - amount
# dataset['amount'] = df.apply(lambda x: convert_currency(x),axis=1)
dataset['amount'] = StandardScaler().fit_transform(dataset['amount'].values.reshape(-1,1))

# Clean - Txn Variant Code
series = df["txvariantcode"].astype('category')
dataset["txvariantcode"] = series.cat.codes
print (dict(enumerate(series.cat.categories))) # {0: 'electron', 1: 'mc', 2: 'mccredit', 3: 'mcdebit', 4: 'visa', 5: 'visabusiness', 6: 'visaclassic', 7: 'visacorporate', 8: 'visadebit', 9: 'visagold', 10: 'visaplatinum', 11: 'visapurchasing', 12: 'visasignature'}

#Clean - Interaction Type
series =  df["shopperinteraction"].astype('category')
dataset['shopperinteraction'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) # {0: 'ContAuth', 1: 'Ecommerce', 2: 'POS'}

#Clean - issue Country
series = df["issuercountrycode"].astype('category')
dataset['issuercountrycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) # Long country map list

#Clean - issuer code - bin
series = df["bin"].astype('category')
dataset['bin'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - currency code
series = df["currencycode"].astype('category')
dataset['currencycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) #0: 'AUD', 1: 'GBP', 2: 'MXN', 3: 'NZD', 4: 'SEK'}

#Clean - Shopper Country code
series = df["shoppercountrycode"].astype('category')
dataset['shoppercountrycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) # Long country map list

#Clean - Card Verif Code Supplied?
series = df["cardverificationcodesupplied"].astype('category')
dataset['cardverificationcodesupplied'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) #{0: False, 1: True}

#Clean - cvcresponse
series = df["cvcresponsecode"].astype('category')
dataset['cvcresponsecode'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) #{0: 0, 1: 1, 2: 2, 3: 3, 4: 5}

# Clean - cvcresponse
# dataset['cvcresponsecode'] = df['cvcresponsecode'].apply(lambda x: (x>2) * 3 + (x<=2)*x)

#Clean - Account code
series = df["accountcode"].astype('category')
dataset['accountcode'] = series.cat.codes
print (dict(enumerate(series.cat.categories))) # {0: 'APACAccount', 1: 'MexicoAccount', 2: 'SwedenAccount', 3: 'UKAccount'}

#Day&Month
dataset['creation_month'] = pd.to_numeric(df.creationdate.dt.month)

# Data IDs
dataset['txnid']    = pd.to_numeric(df["txid"])#int(float(df["txid"]))
dataset['mail_id']  = pd.to_numeric(df["mail_id"].str.replace('email','').str.replace('NA','0'))#int(float(df["mail_id"].replace('email','')))
dataset['ip_id']    = pd.to_numeric(df["ip_id"].str.replace('ip','').str.replace('NA','0'))#int(float(df["ip_id"].replace('ip','')))
dataset['card_id']  = pd.to_numeric(df["card_id"].str.replace('card','').str.replace('NA','0'))#int(float(df["card_id"].replace('card','')))

# Clean - Simple Journel as label
series = df['simple_journal'].apply(lambda x: x=='Chargeback').astype('category') # assign labels: 1 for fraud, 0 for normal
dataset['label'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

# Dataframe to numpy array
data_array = dataset.as_matrix()

# Complete Data set Save into CSV
np.savetxt("cleaned_data.csv", data_array, delimiter=",")


# Take 2 with DF
categorical=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
               'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']
for col in categorical:
    df[col] = df[col].astype('category')

df['convertedAmount'] = df.apply(lambda x: convert_currency(x),axis=1)
df['convertedAmount'] = StandardScaler().fit_transform(df['amount'].values.reshape(-1,1))

df.to_csv(index=False,path_or_buf=os.getcwd()+'/df_cleaned.csv')

#Data Explore - Navin
# fraud_data  = df_clean[df_clean.label == 1]
# benign_data = df_clean[df_clean.label == 0]
#
# print(np.mean(fraud_data.amount))
# print(np.mean(benign_data.amount))
#
# print(np.max(fraud_data.amount))
# print(np.max(benign_data.amount))
#
# print(df_clean.keys())
# print(df_clean["cvcresponsecode"].unique())
# sns.boxplot(x=df_clean["label"], y=df_clean["cvcresponsecode"])
# sns.boxplot(x=df_clean["label"], y=df_clean["amount"])





