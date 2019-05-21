import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ARMA Implementation - Basic Model
# 1. At-least 5 individual sensible sensors.
# 2. Auto-Correl plot to iderntify order of ARMA
# 3. Determine params - AIC
# 4. Select Threshold for detection - reason
# 5. Study Detected Anamolies
# 6. What kind of Analmolies can ARMA model?
# 7. Which Signals are good?

train_dataset = pd.read_csv('dataset/training_1.csv')
tune_dataset  = pd.read_csv('dataset/training_2.csv')
test_dataset  = pd.read_csv('dataset/testing.csv')

print(train_dataset.columns.values)
print(tune_dataset.columns.values)
print(test_dataset.columns.values)
