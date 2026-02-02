import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("insurance.csv")
print(df.shape)
print(df.head())

#Class Distribution
plt.figure(figsize=(6,4))
df['insurance'].value_counts().plot(kind='bar')
plt.title("Target Class Distribution")
plt.xlabel("Insurance Class")
plt.ylabel("Count")
plt.show()