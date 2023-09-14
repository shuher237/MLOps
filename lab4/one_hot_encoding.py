import numpy as np
import pandas as pd
import os

#path variable
path = './datasets/titanic_df.csv'

df = pd.read_csv(path)
df[['Male', 'Female']] = pd.get_dummies(df['Sex'])
df.to_csv(path)

print(df.head(5))
