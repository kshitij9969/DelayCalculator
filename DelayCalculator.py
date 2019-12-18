#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:22 2019

@author: kshitijsingh
"""


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import os
import seaborn as sns
from math import sqrt
py.init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')
os.chdir(r"/Users/kshitijsingh/Downloads/DelayCalculator") # For macOS


# Importing the dataset
dataset = pd.read_excel('Corrected_dataset.xlsx')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#### Neural Network ####


sns.pairplot(dataset)

# Scaling the features
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
X= sc.fit_transform(X)
y= y.reshape(-1,1)
y=sc.fit_transform(y)


# Creating the model
from keras import Sequential
from keras.layers import Dense
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=2, input_dim=2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor


from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=100,epochs=100)


results=regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
# from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 1)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)



# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
print(explained_variance_score(y_test, y_pred))
max_error(y_test, y_pred)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

print(sqrt(mean_squared_error))
print(median_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
mean_poisson_deviance(y_test, y_pred)
mean_gamma_deviance(y_test,y_pred)





# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()