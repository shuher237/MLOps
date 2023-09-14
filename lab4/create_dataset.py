import numpy as np
import pandas as pd
import os
from catboost.datasets import titanic


titanic_train, titanic_test = titanic()

titanic_df = pd.concat([titanic_train, titanic_test])
titanic_df.to_csv('./datasets/titanic_df.csv')
print(titanic_df.head(5))
 
