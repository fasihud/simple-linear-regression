#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:34:08 2019

@author: void
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[: , :-1].values

Y = dataset.iloc[: , 1].values

# For missing Values

#from sklearn.preprocessing import Imputer
#
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[: , 1:3])
#X[: , 1:3] = imputer.transform(X[: , 1:3])

# Encoding categorical data

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#le_X = LabelEncoder()
#X[: , 0] = le_X.fit_transform(X[: , 0])
#
#le_Y = LabelEncoder()
#Y = le_Y.fit_transform(Y)
#
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

# Splitting Data into Train and Test Sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Simple Linear Regression in the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting Test Set Data
y_pred = regressor.predict(X_test)

# Visualizing Training Sets Results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()


# Visualizing Test Sets Results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()









