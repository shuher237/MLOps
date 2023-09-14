import numpy as np
import pandas as pd
import os

#path variable
path = './datasets/titanic_df.csv'

#fillna 'Age' column
df = pd.read_csv(path)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.to_csv(path)

print(df['Age'].mean())
