import os

import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')


class DataCleaner():

    def trigger_clean(self, src):

        df = pd.read_csv(src)

        df = df[df['simple_journal']!='Refused']

        df['creationdate'] = pd.to_datetime(df['creationdate'],format='%Y-%m-%d %H:%M:%S')

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

        # Data Clean
        categorical=['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
                       'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']
        for col in categorical:
            df[col] = df[col].astype('category')

        df['convertedAmount'] = df.apply(lambda x: convert_currency(x),axis=1)
        df['convertedAmount'] = StandardScaler().fit_transform(df['amount'].values.reshape(-1,1))

        df.to_csv(index=False,path_or_buf=os.getcwd()+'/df_cleaned.csv')
