# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:45:09 2021

@author: Ziad
"""
# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,date,timedelta


# feture to be predicted 
feature = 'SalesAmount'
fulldata = pd.read_csv('C:\\MyProjects\\Roche\Data\\YelloCabDataModel.csv', parse_dates=['date']).fillna(0)
#fulldata = pd.read_csv('C:\\MyProjects\\Roche\Data\\YelloCabDataModel.csv', parse_dates=['date']).fillna(0)

df1=fulldata[(fulldata['date'] >= '2020-03-15') ]

# plt.figure(figsize=(12,5))
# plt.plot(df['date'], df[feature]) # 'r' is the color red
# plt.xlabel('date')
# plt.ylabel('Taxi Sales')
# plt.title('Taxi Sales ')
# plt.show()
#df2= df[feature].tolist()
#plt.plot(df2)
#df.groupby(['date'])['SalesAmount','Cases'].sum().plot(figsize=(14,7),color=['b','r'])


df1=df1.reset_index()[feature]
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data=df1[0:training_size,:]
train_data = df1[:,:]
test_data =df1[training_size:len(df1),:1]
training_size,test_size
# change array into Matrix of len of trainig data by times_steps
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_steps = 30
X_train, y_train = create_dataset(train_data, time_steps)
X_test, ytest = create_dataset(test_data, time_steps)
print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
# reshape input to be [samples, time steps, features] which is required for LSTM

# Building the RNN Model 
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model=Sequential()
# Add LSTM layer
model.add(LSTM(50,return_sequences=True,input_shape=(time_steps,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='rmsprop')
#model.compile(loss='mean_squared_error',optimizer='rmsprop')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=10,verbose=1)
print(model.summary())


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
### Calculate RMSE performance metrics

import math
from sklearn.metrics import mean_squared_error
#print("Mean Sequared Error : ",math.sqrt(mean_squared_error(y_train,train_predict)))
### Test Data RMSE
print("Mean Sequared Error : ",math.sqrt(mean_squared_error(ytest,test_predict)))

# Predicting the future time series result for the next number of days
x_input=test_data[len(test_data) -time_steps:].reshape(1,-1)
x_input.shape
# demonstrate prediction for next 180 days
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
#***************
lst_output=[]
n_steps=time_steps
i=0
future_days = 365
while(i<future_days):
     if(len(temp_input)> n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input.shape)
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat)
        i=i+1
     else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

df1=scaler.inverse_transform(np.array(df1).reshape(-1,1))
lst_output = scaler.inverse_transform(lst_output)
## Plot next Future dayes
plt.figure(figsize=(12,5))
y2 = list(lst_output.reshape(-1))
x2 = list(range(len(df1)+1,len(df1)+len(lst_output)+1))
plt.plot(x2,y2)

plt.figure(figsize=(12,5))
y1 = df1
x1 = list(range(1,len(df1)+1))
y2 = list(lst_output.reshape(-1))
x2 = list(range(len(df1)+1,len(df1)+len(lst_output)+1))
plt.plot(x1, y1,x2,y2)

sdate = fulldata['date'].max() + timedelta(days=1) 
edate = sdate + timedelta(days=future_days)
def dates_bwn_twodates(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)        
date_list=[]
for d in dates_bwn_twodates(sdate,edate):
    date_list.append( d)

data_tuples = list(zip(date_list,list(lst_output.reshape(-1))))
df4 = pd.DataFrame(data_tuples, columns=['date',feature])   
fileaname ="Predicted" + feature + "forNext" + str(future_days) + ".csv"
df4.to_csv('C:\\MyProjects\\Roche\Data\\'+fileaname )

#df4 = pd.DataFrame({'date': date_list, 'SalesAmount': lst_output})
df5 = pd.concat([fulldata, df4], axis=0)
fileaname ="FullDataPrediction" + feature + "forNext" + str(future_days) + ".csv"
df5.to_csv('C:\\MyProjects\\Roche\Data\\' + fileaname)

#df5.groupby(['date'])[feature].sum().plot(figsize=(18,7),color=['b','r'])
