#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kshitijsingh

Dataset Description: 
A. Dataset contains three columns viz.
    1. N        -      Number of TCLs(int)
    2. Pnorm(%) -      Normalized % power of each TCL.(float)
    3. Alpha    -      Delay(float)
B.  Length - 49490
"""


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.offline as py
import os
import seaborn as sns
from math import sqrt
from keras import Sequential
from keras.layers import Dense

py.init_notebook_mode(connected=True)
plt.rc('text', 
       usetex=True)
plt.style.use('fivethirtyeight')
os.chdir(r"%%Path to the working directory") # For macOS
os.chdir(r"%%Path to working directory") # For windows


''' 
Importing the dataset:
    Here we import the dataset and split it into 
    matrix of features(independent variabsles)
    and dependent(target) variables
    Variables:
        1. NP - Holds columns N and P
        2. a - Holds column alpha
'''

dataset = pd.read_excel('/Users/kshitijsingh/Downloads/temp1.xlsx')
NP = dataset.iloc[:, 0:2].values
a = dataset.iloc[:, 2].values

''' Visualising the dataset '''
# The whole dataset
plt.scatter(a,
            NP[:,0:1])
plt.rc('text', 
       usetex=True)
plt.ylabel(r'$P_{norm}$(%)',
           fontsize=24)
plt.xlabel(r'$\alpha$', 
           fontsize=24)
plt.legend()
plt.title(r'Variation of $P_{norm}$(%) vs $\alpha$ as N varies from 10 to 500')

# Visualising P vs alpha
plt.scatter(a[0:100],
            NP[0:100,1], 
            label='N = 10') # For N = 10
plt.scatter(a[500:600],
            NP[500:600,1], 
            label='N = 15') # For N = 15
plt.scatter(a[1000:1100],
            NP[1000:1100,1], 
            label='N = 20') # For N = 20
plt.scatter(a[2000:2100],
            NP[2000:2100,1], 
            label='N = 30') # For N = 30
plt.legend()
plt.title(r'$P_{norm}$(%) vs $\alpha$')
plt.ylabel(r'$P_{norm}$(%)',
           fontsize=24)
plt.xlabel(r'$\alpha$', 
           fontsize=24)


''' Creating the neural network model '''
# Scaling the features
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
NP= sc.fit_transform(NP)
a= a.reshape(-1,1)
a=sc.fit_transform(a)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
NP_train, NP_test, a_train, a_test = train_test_split(NP, 
                                                      a, 
                                                      test_size = 0.3, 
                                                      random_state = 0)

# Creating the model
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=2, 
                        input_dim=2))
    regressor.add(Dense(units=4, 
                        activation='relu'))    
    regressor.add(Dense(units=1, 
                        activation='sigmoid'))
    regressor.compile(optimizer='adam', 
                      loss='mean_squared_error',  
                      metrics=['mse','accuracy'])
    return regressor

# Building the model
from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor,
                           batch_size=64,
                           epochs=50)

# Training and prediction
results=regressor.fit(NP_train,a_train)
a_pred= regressor.predict(NP_test)


''' Visualising the results '''

# Line plot comparision
plt.plot(a_pred[0:100], 
         color = 'red', 
         label=r'$\alpha_{pred}$')
plt.plot(a_test[0:100], 
         color ='dodgerblue', 
         label=r'$\alpha_{true}$')
plt.plot(a_pred[0:100], 
         'g*')
plt.ylabel(r'$\alpha$', 
           fontsize=24)
plt.xlabel(r'Time Series',
           fontsize=24)
plt.title(r'$\alpha_{pred}$ vs $\alpha_{test}$',
          fontsize = 48)
plt.legend()
plt.show()

# Scatter plot of predicted values vs true values
plt.scatter(a_pred,
            a_test)
plt.xlabel(r'$\alpha_{pred}$(°)', 
           fontsize = 18)
plt.ylabel(r'$\alpha_{test}$(°)', 
           fontsize = 18)
plt.title(r'$\alpha_{pred}$ vs $\alpha_{test}$', 
          fontsize = 48)

# Plotting the error lines
x = np.linspace(start=0,
                stop = 0.8,
                num = 100)
plt.plot(x,
         x,
         color ='black', 
         dashes=[5,5])
plt.plot(x, 
         x+(0.1), 
         color ='red', 
         dashes=[3,3])
plt.plot(x, 
         x-(0.1), 
         color ='red', 
         dashes=[3,3])

# Plotting loss(MSE) vs epochs
ep = np.linspace(start=1, 
                 stop=50, 
                 num = 50)
plt.plot(ep,
         results.history['loss'], 
         color='dodgerblue', 
         label='Loss')
plt.xlabel(r'Epochs', 
           fontsize = 18)
plt.ylabel(r'Loss', 
           fontsize = 18)
plt.legend()

# Plotting MSE and RMSE vs epochs
ep = np.linspace(start=1,
                 stop=50, 
                 num = 50)
plt.plot(ep,
         results.history['mse'], 
         color='dodgerblue', 
         label='Mean Squared Error')
plt.xlabel(r'Epochs', 
           fontsize = 18)
plt.ylabel(r'Mean Squared Error', 
           fontsize = 18)
ep = np.linspace(start=1, 
                 stop=50, 
                 num = 50)
RMSE = np.sqrt(results.history['mse'])
plt.plot(ep,
         RMSE,
         color='dodgerblue', 
         label='Root Mean Squared Error')
plt.legend()
plt.xlabel(r'Epochs', 
           fontsize = 18)
plt.ylabel(r'Root Mean Squared Error', 
           fontsize = 18)


''' Metrics '''
# Evaluating the performance of model
from sklearn.metrics import explained_variance_score, max_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

print(explained_variance_score(a_test, a_pred))
print(max_error(a_test, a_pred))
print(mean_absolute_error(a_test, a_pred))
print(mean_squared_error(a_test, a_pred))
print(sqrt(mean_squared_error(a_test, a_pred)))
print(median_absolute_error(a_test, a_pred))
print(r2_score(a_test, a_pred))
