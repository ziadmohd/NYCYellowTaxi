# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:41:31 2021

@author: Ziad
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:26:27 2019

@author: zm40717
Black Friday 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score

import xgboost 


dataset = pd.read_csv("C:\MyProjects\Roche\Data\YelloCabDataModel.csv").fillna(0)
dataset= dataset[dataset['date'] > int('20200301')]

#dataset = pd.read_csv("C:\MyProjects\Roche\Data\FullDataPredictionSalesAmountforNext365.csv").fillna(0)
#dataset= dataset[dataset['date'] > ('20200301')]
# Feature Scaling
X = dataset.iloc[:, -1].values
y = dataset.iloc[:, 1:2].values
y =np.array(y).reshape(-1).astype(int)
X= np.array(X.reshape(-1,1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
plt.figure(figsize=(10,5))
plt.xlabel("Cases")
plt.ylabel("Sales")
plt.scatter(X , y)

from sklearn.linear_model import LinearRegression
# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure(figsize=(12,5))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('4 degree Polynomial Regression')
plt.xlabel('Cases')
plt.ylabel('Sales')
plt.show()
### Calculate the polynomial score
y_pred = np.array(lin_reg_2.predict(poly_reg.fit_transform(X_test))).astype(int)
y_train_pred = np.array(lin_reg_2.predict(poly_reg.fit_transform(Y_train.reshape(-1,1)))).astype(int)
# Polynomial Regression
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results
accuracy = polyfit(y_test,y_pred,4)
print(accuracy['determination'])
model_score = dict({"Model":[] ,"Accuracy":[],"Notes":[]})
model_score["Model"].append("Polynomial")
model_score["Accuracy"].append(accuracy['determination']*100)
model_score["Notes"].append("Bias")
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print(rmse)
#print("Test Data r2: ",r2_score(y_test, y_pred)*100)
#print("Train Data r2: ",r2_score(X_test, y_train_pred)*100)


# print(lin_reg_2.coef_)
# inp = np.array(1000).reshape(-1,1)
# print((lin_reg_2.predict(poly_reg.fit_transform(inp))))


# SVC
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVR
X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_y.fit_transform(np.array(Y_train).reshape(-1,1))
regressor = SVR(kernel = 'rbf',gamma=0.001,C=100)
regressor.fit(X_train, Y_train.reshape(-1))
#Y_train = np.array(Y_train.reshape(-1))
y_test =np.array(y_test).reshape(-1,1).astype(int)
y_pred = sc_y.inverse_transform(np.array(regressor.predict(X_test)))
accuracy= r2_score(y_test,y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
model_score["Model"].append("SVM")
model_score["Accuracy"].append(accuracy*100)
model_score["Notes"].append("Accuracy overfit")

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 10)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
f = sc_y.inverse_transform(regressor.predict(X_grid))
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(X_grid)), color = 'blue')
plt.title('Cases vs Sales')
plt.xlabel('Cases')
plt.ylabel('Salary')
plt.show()


# # GaussianNB
# # Result is over fit model 
# regressor = GaussianNB()
# regressor.fit(X_train, Y_train)
# y_pred = np.array(regressor.predict(X_test)).reshape(-1,1)

# accuracy = r2_score(y_test, y_pred)

# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# model_score["Model"].append("Gaussian")
# model_score["Accuracy"].append(accuracy*100)
# model_score["Notes"].append("High Accuracy overfit")


# #xgboost  Fitting XGBoost to the Training set
# from xgboost import XGBClassifier
# X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# # sc_X = StandardScaler()
# # sc_y = StandardScaler()
# # sc_y_test = StandardScaler()
# # X_train = sc_X.fit_transform(X_train)
# # Y_train = sc_y.fit_transform(np.array(Y_train).reshape(-1,1))
# # X_test = sc_y_test.fit_transform(np.array(X_test).reshape(-1,1))

# regressor = XGBClassifier(max_depth=3)
# regressor.fit(X_train, X_train.reshape(-1))
# y_pred = regressor.predict(X_test)

# #accuracy = accuracy_score(y_test, y_pred)
# accuracy = r2_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# model_score["Model"].append("xgboost")
# model_score["Accuracy"].append(accuracy*100)
# model_score["Notes"].append(" Accuracy underfit")


# # ANN Importing the Keras libraries and packages
# #import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(-1,1))

# X_train1 = scaler.fit_transform(X_train)
# Y_train1 = scaler.fit_transform(np.array(Y_train.reshape(-1,1)))
# X_test1 = scaler.fit_transform(X_test)

# #Initialising the ANN
# ann_r = Sequential()
# # Adding the input layer and the first hidden layer
# ann_r.add(Dense(units = 6, kernel_initializer = 'normal', activation = 'relu', input_dim = 1))
# # Adding the second hidden layer
# ann_r.add(Dense(units = 6, kernel_initializer = 'normal', activation = 'relu'))
# # Adding the output layer
# ann_r.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'relu'))
# # Compiling the ANN
# ann_r.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
# # Fitting the ANN to the Training set
# ann_r.fit(X_train1, Y_train1, batch_size = 10, epochs = 200)
# # Part 3 - Making predictions and evaluating the model
# # Predicting the Test set results
# y_pred1 = scaler.inverse_transform( ann_r.predict(X_test1)).astype(int)
# #y_pred1 = np.array(y_pred.reshape(-1))
# _, accuracy = ann_r.evaluate(y_test, y_pred1)
# accuracy = accuracy_score(y_test, y_pred1)
# print('Accuracy: %.2f' % (accuracy*100))
# model_score["Model"].append("ANN")
# model_score["Accuracy"].append(accuracy*100)
# model_score["Notes"].append("Zero Accuracy")

