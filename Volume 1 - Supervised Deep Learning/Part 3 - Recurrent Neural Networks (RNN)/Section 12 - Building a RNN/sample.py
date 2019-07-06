# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:11:41 2019

@author: Shubham
"""

#RNN

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#features scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#building RNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#initializing
regressor = Sequential()

#first lstm and dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(rate= 0.2))

#second lstm and dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.2))

#third lstm and dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(rate= 0.2))

#fourth lstm and dropout regularisation
regressor.add(LSTM(units= 50))
regressor.add(Dropout(rate= 0.2))

#output layer
regressor.add(Dense(units= 1))

#compiling the RNN
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')

#fitting rnn
regressor.fit(X_train, y_train, batch_size= 32, epochs= 100)

#getting real stock price
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#predicting
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

