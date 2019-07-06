# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:38:03 2019

@author: Shubham
"""

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[:, -1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

#fitting som
from minisom import MiniSom
som = MiniSom(x=10, y=10,
              input_len=15, 
              sigma=1.0,
              learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#visualising the som
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
marker = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         marker[y[i]],
         markeredgecolor= colors[y[i]],
         markerfacecolor= 'None',
         markersize= 10,
         markeredgewidth= 2)
show()

#finding fraouds
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(8,1)], mapping[(6,6)]), axis= 0)
frauds = sc.inverse_transform(frauds)