import numpy as np
import pandas as pd
import pickle

#lib below for preprocessing and testing
from sklearn.preprocessing import MinMaxScaler
from data_creation import check_folder

#path variables
train_input_path_1 = './train/train_sample_1.csv'
train_input_path_2 = './train/train_sample_2.csv'

test_input_path_1 = './test/test_sample_1.csv'
test_input_path_2 = './test/test_sample_2.csv'

output_path = './after_preprocessing/'


def create_dataset(input_path_1: str, input_path_2: str, dataset_name, output_path: str):

    #read samples
    ds_1 = pd.read_csv(input_path_1, index_col=0)
    ds_1 = ds_1.sort_values(by='Date', ascending=True)

    ds_2 = pd.read_csv(input_path_2, index_col=0)
    ds_2 = ds_2.sort_values(by='Date', ascending=True)
    
    ds = ds_1.merge(ds_2, how='inner', on='Date')

    #merge samples
    ds['temp'] = (ds['Temp'] + ds['Temperature'])/2
    ds_result = ds['temp']
    
    #normalize the dataset
    scaler = MinMaxScaler()
    dataset = ds_result.values
    dataset = dataset.reshape(-1,1)
    dataset = scaler.fit_transform(dataset)
    dataset = dataset.astype('float32')

    check_folder(output_path)
    np.save(f'{output_path}{dataset_name}.npy', dataset)

    #save scaler
    with open('./scaler.pkl', 'wb') as output:
        pickle.dump(scaler, output)



if __name__ == '__main__':
    create_dataset(train_input_path_1, train_input_path_2, 'train', output_path)
    create_dataset(test_input_path_1, test_input_path_2, 'test', output_path)