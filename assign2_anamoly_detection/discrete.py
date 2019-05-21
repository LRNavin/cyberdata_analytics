import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Discrete Implementation
# 1. Discretize dataset - Use any method - Why choose? & Why makes sense?
# 2. Visualize Discretion
# 3. Find Anamolies - N-grams or kNN to sliding window lenghts (choose length)
# 4. If prob less (N-gram) or too far data (kNN) - Raise alarm - Select Threshold for detection - reason
# 5. What kind of Analmolies can DISCRETE model?
# 6. Which Signals are good?

train_dataset = pd.read_csv('dataset/training_1.csv')
tune_dataset  = pd.read_csv('dataset/training_2.csv')
test_dataset  = pd.read_csv('dataset/testing.csv')

print(train_dataset.columns.values)
print(tune_dataset.columns.values)
print(test_dataset.columns.values)
