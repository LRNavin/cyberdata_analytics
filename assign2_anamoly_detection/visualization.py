import seaborn as sns
import scipy as sp
from pandas import DataFrame as df
import pandas as pd

src = 'data/'
starting_df = pd.read_csv(src+'BATADAL_dataset03.csv')
optimizing_df = pd.read_csv(src+'BATADAL_dataset04.csv')
testing_df = pd.read_csv(src+'BATADAL_test_dataset.csv')

print(starting_df.head())
