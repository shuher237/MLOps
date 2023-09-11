import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle
import keras
from model_preparation import create_series

test_dataset_path = './after_preprocessing/test.npy'
test = np.load(test_dataset_path)

testX, testY = create_series(test, look_back=1)

if __name__ == '__main__':

    model = keras.models.load_model('./model.keras')
    
    with open('./scaler.pkl', 'rb') as pkl_file_1:
        scaler = pickle.load(pkl_file_1)
    # make predictions
    testPredict = model.predict(testX)
    
    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    # calculate root mean squared error
    RMSE_Score = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (RMSE_Score))
    MAE_Score = mean_absolute_error(testY[0], testPredict[:,0])
    print('Test Score: %.2f MAE' % (MAE_Score))
    MPAE_Score = mean_absolute_percentage_error(testY[0], testPredict[:,0])
    print('Test Score: %.2f MPAE' % (MPAE_Score))

    print('This is jenkins integration')
