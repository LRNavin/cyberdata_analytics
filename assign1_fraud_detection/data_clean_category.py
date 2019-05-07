# import packages
# from google.colab import drive

import seaborn as sns
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

plt.style.use('ggplot')


#Data Explore
# src = 'data_for_student_case.csv'
# drive.mount('/content/drive')

src = 'adyen_data.csv'
df = pd.read_csv(src)

# df_clean = df.copy()
df = df[df['simple_journal']!='Refused']

# df_clean['creation_date'] = pd.to_datetime(df_clean['creationdate'],format='%Y-%m-%d %H:%M:%S')
# df_clean['creation_month'] = df_clean.creation_date.dt.month
# df_clean['creation_weekday'] = df_clean.creation_date.dt.weekday
# df_clean['creation_day'] = df_clean.creation_date.dt.date

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
dataset['amount'] = df.apply(lambda x: convert_currency(x),axis=1)

# Clean - Txn Variant Code
series = df["txvariantcode"].astype('category')
dataset["txvariantcode"] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - Interaction Type
series =  df["shopperinteraction"].astype('category')
dataset['shopperinteraction'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - issue Country
series = df["issuercountrycode"].astype('category')
dataset['issuercountrycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - issuer code - bin
dataset['bin'] = df["bin"]

#Clean - currency code
series = df["currencycode"].astype('category')
dataset['currencycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - Shopper Country code
series = df["shoppercountrycode"].astype('category')
dataset['shoppercountrycode'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - Card Verif Code Supplied?
series = df["cardverificationcodesupplied"].astype('category')
dataset['cardverificationcodesupplied'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

#Clean - cvcresponse
series = df["cvcresponsecode"].astype('category')
dataset['cvcresponsecode'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

# Clean - cvcresponse
# dataset['cvcresponsecode'] = df['cvcresponsecode'].apply(lambda x: (x>2) * 3 + (x<=2)*x)

#Clean - Account code
series = df["accountcode"].astype('category')
dataset['accountcode'] = series.cat.codes
print (dict(enumerate(series.cat.categories)))

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





