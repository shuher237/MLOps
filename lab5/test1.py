
import pytest
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

output_folder = "./"

@pytest.fixture()
def init_data():
  
  vec11 = np.load(f'{output_folder}X_s_2.npy')
  vec12 = np.load(f'{output_folder}Y_s_2.npy')
  vec21 = np.load(f'{output_folder}X_s_3.npy')
  vec22 = np.load(f'{output_folder}Y_s_3.npy')
  vec31 = np.load(f'{output_folder}X_s_err.npy')
  vec32 = np.load(f'{output_folder}Y_s_err.npy')
  
  return vec11, vec12, vec21, vec22, vec31, vec32

def test1(init_data):
  with open(f'{output_folder}myfile.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)
    y_pred = regressor.predict(init_data[0].reshape(-1, 1))
    assert mean_absolute_error(init_data[1], y_pred) <= 2

def test2(init_data):
  with open(f'{output_folder}myfile.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)
    y_pred = regressor.predict(init_data[2].reshape(-1, 1))
    assert mean_absolute_error(init_data[3], y_pred) <= 2

def test3(init_data):
  with open(f'{output_folder}myfile.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)
    y_pred = regressor.predict(init_data[4].reshape(-1, 1))
    assert mean_absolute_error(init_data[5], y_pred) <= 2
