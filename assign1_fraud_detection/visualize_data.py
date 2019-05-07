
import seaborn as sns
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

sns.set(color_codes=True)

plt.style.use('ggplot')

src = 'cleaned_data.csv'
dataset = pd.read_csv(src).as_matrix()

# dataset = dataset[dataset[:,-1] == 0]

x = dataset[:,:-1]
y = dataset[:,-1]


######################## Dataset column reference with index ########################

# 0 . Amount
# 1 . Txn Variant Code
# 2 . Interaction Type
# 3 . issue Country
# 4 . issuer code - bin
# 5 . currency code
# 6 . Shopper Country code
# 7 . Card Verif Code Supplied?
# 8 . cvcresponse
# 9 . Account code
# 10 . txnid
# 11 . mail_id
# 12 . ip_id
# 13 . card_id
# 14 . simple_journal (class label)

######################## ################################## ########################


#1. amount histogram fraudulent transactions and non fraudulent transactions
sns.countplot(y)
plt.show()


#2. where fraudulent transactions are most prevailant: shopperinteraction
sns.countplot(x[:,2])
plt.show()

#3. difference in fraudulent and non transactions per shopper country code
sns.countplot(x=y, hue=x[:,6])
plt.show()

#4. difference in fraudulent and non transactions per currency code
sns.countplot(x=y, hue=x[:,5])
plt.show()

#5. cvc response code vs amount of money for fraud and non fraud
sns.boxplot(x=y, y=x[:,0])
plt.show()

#6. difference visa and mastercard for fraud and non fraud transactions
sns.countplot(x=y, hue=x[:,8])
plt.show()

# Count plot CVC Response Vs Fraud/Non-Fraud
sns.countplot(x=y, hue=x[:,1])
plt.show()
