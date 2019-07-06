# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:37:50 2019

@author: Shubham
"""
#ANN 

#part1 (data preprocessing)
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,13].values

#categories
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
labelencoder_X1 = LabelEncoder()
X[:,2] = labelencoder_X1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1], categories=None)
X = onehotencoder.fit_transform(X).toarray()
#1 col france 2 col ger snd 3 col spain
X = X[:,1:] #france col removed to avoid dummy variable trap

#spliting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#part 2 ANN building

#lib and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ann
classifier = Sequential()

#input and first hidden layer
classifier.add(Dense(units=6 ,activation='relu' ,input_dim=11))

#second hidden layer
classifier.add(Dense(units=6,activation='relu'))

#output layer
classifier.add(Dense(units=1,activation='sigmoid'))

#compiling ann
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

#fitting ann to training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)


#part 3
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)