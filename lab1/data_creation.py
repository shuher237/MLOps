import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import os

def check_folder(folder):
    isexist = os.path.exists(folder)
    if not isexist:
        os.makedirs(folder)


if __name__ == '__main__':
    #reading datasets
    url_1 = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    response_1 = requests.get(url_1)
    data = StringIO(response_1.text)
    temp_df_1 = pd.read_csv(data)


    url_2 = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv'
    response_2 = requests.get(url_2)
    data = StringIO(response_2.text)
    temp_df_2 = pd.read_csv(data)


    #dividing to samples
    train_sample_1, test_sample_1 = train_test_split(temp_df_1, test_size=0.3, random_state=13)
    train_sample_2, test_sample_2 = train_test_split(temp_df_2, test_size=0.3, random_state=13)

    folder = './train/'
    check_folder(folder)
    train_sample_1.to_csv(f'{folder}train_sample_1.csv')
    train_sample_2.to_csv(f'{folder}train_sample_2.csv')
    
    folder = './test/'
    check_folder(folder)
    test_sample_1.to_csv(f'{folder}test_sample_1.csv')
    test_sample_2.to_csv(f'{folder}test_sample_2.csv')