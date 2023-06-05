import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')

from time import time

import matplotlib.ticker as tkr

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
    
import math

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import datetime







# Function to generate the next date excluding weekends
def generate_next_date(current_date):
    next_date = current_date + datetime.timedelta(days=1)
    while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_date += datetime.timedelta(days=1)
    return next_date



def convert_columns_to_rows(df):
    # Convert index to datetime if needed
    df.index = pd.to_datetime(df.index)


    # Get the current date from the DataFrame index
    current_date = df.index[0]

    # Create an empty DataFrame to store the converted data
    new_df = pd.DataFrame(columns=['date', 'pred+0'])

    # Iterate over each column and convert values into rows
    for column in df.columns:
        current_value = df[column][0]

        # Generate the next date excluding weekends
        next_date = generate_next_date(current_date)

        # Add the current date and value as a row in the new DataFrame
        new_row = pd.DataFrame({'date': [current_date], 'pred+0': [current_value]})
        new_df = pd.concat([new_df, new_row], ignore_index=True)

        # Update the current date for the next iteration
        current_date = next_date

    new_df.set_index('date', inplace=True)
    return new_df










def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def simpleTimeSeriesPred_LSTM(dataset, look_back, number_of_dense_layer=1):
    
    # original time serie (Y)
    y = dataset.close.values 
    y = y.astype('float32')
    y = np.reshape(y, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(y)



    # training and testing settings (size)
    percent_of_training = 0.7
    train_size = int(len(y) * percent_of_training)
    test_size = len(y) - train_size
    # 
    train_y, test_y = y[0:train_size,:], y[train_size:len(y),:]    
    
    
    # features of the original time serie (y)
    X_train_features_1, y_train = create_dataset(train_y, look_back)
    X_test_features_1, y_test = create_dataset(test_y, look_back)




    # join the all the features in one
    ## reshape arrays
    X_train_features = np.reshape(X_train_features_1, (X_train_features_1.shape[0], 1, X_train_features_1.shape[1]))
    X_test_features  = np.reshape(X_test_features_1, (X_test_features_1.shape[0], 1, X_test_features_1.shape[1]))
    
    
    
    #define MODEL
    model = Sequential()
    model.add(LSTM(200, input_shape=(X_train_features.shape[1], X_train_features.shape[2])))
    model.add(Dropout(0.20))
    model.add(Dense(number_of_dense_layer))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train_features,y_train, epochs=300, batch_size=25, validation_data=(X_test_features, y_test), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=False)

    print(model.summary())
    
    
    
    #Train, Test, Predict
    
    train_predict = model.predict(X_train_features)
    test_predict  = model.predict(X_test_features)



    #train_predict = scaler.inverse_transform(train_predict)
    #Y_train = scaler.inverse_transform(y_train)
    #test_predict = scaler.inverse_transform(test_predict)
    #Y_test = scaler.inverse_transform(y_test)


    print('Train Mean Absolute Error:', mean_absolute_error(np.reshape(y_train,(y_train.shape[0],1)), train_predict[:,0]))
    print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(np.reshape(y_train,(y_train.shape[0],1)), train_predict[:,0])))
    print('Test Mean Absolute Error:', mean_absolute_error(np.reshape(y_test,(y_test.shape[0],1)), test_predict[:,0]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(np.reshape(y_test,(y_test.shape[0],1)), test_predict[:,0])))
    
    train_predict= scaler.inverse_transform(train_predict)
    test_predict= scaler.inverse_transform(test_predict)
    
    
    columns_name =[]
    for i in range(number_of_dense_layer):
        columns_name.append(f"pred+{i}")
        
    
    time_y_train = pd.DataFrame(data = train_y, index = dataset[0:train_size].index,columns= [""])
    time_y_test  = pd.DataFrame(data = test_y, index = dataset[train_size:].index,columns= [""])
    time_y_train_prediction = pd.DataFrame(data = train_predict, index = time_y_train[look_back+1:].index,columns= columns_name)
    time_y_test_prediction  = pd.DataFrame(data = test_predict, index = time_y_test[look_back+1:].index,columns= columns_name)
    
    
    
    predections = convert_columns_to_rows(time_y_test_prediction[-1:])
    
    
    
        
    return predections ,time_y_train_prediction, time_y_test_prediction , model