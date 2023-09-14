import numpy as np
import pandas as pd
import os

#path variable
path = './datasets/titanic_df.csv'


#modify dataset
df = pd.read_csv(path)
df.drop(df.columns[0], axis=1, inplace= True)
df.to_csv(path)

print(df.columns.values.tolist())

