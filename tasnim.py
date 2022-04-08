# -*- coding: utf-8 -*-
"""
Inam Maraqa
Tasnim Alkaraki
Rawand Yasin

"""
import pandas as pd
import numpy as np
from tkinter import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


data = pd.read_csv('2010-capitalbikeshare-tripdata.csv')
data = data.loc[:, ['Start date','Start station number']]

data[['date','time']]= data['Start date'].str.split(expand=True)
#print (data['date'])

data['date'] = data.date.str.split('-').str[2].astype(int)
data['time'] = data.time.str.split(':').str[0].astype(int)
print (data['date'])

data = pd.DataFrame({'count' : data.groupby(['Start station number','date','time'] ).size()}).reset_index()

features = data[['Start station number', 'date', 'time']]
target = data[['count']]





plt.scatter(features['Start station number'] , target['count'])
plt.xlabel("Start station number")
plt.ylabel("count")
plt.show()

plt.scatter(features['date'] , target['count'])
plt.xlabel("date")
plt.ylabel("count")
plt.show()

plt.scatter(features['time'] , target['count'])
plt.xlabel("time")
plt.ylabel("count")
plt.show()


scaler = MinMaxScaler()
scaler.fit(features)
features=scaler.transform(features)


#LinearRegression
reg = LinearRegression()
reg.fit(features, target)
reg.score(features, target)

k =KFold(n_splits=5) 
pred = cross_val_score(reg,features, target, cv=k, scoring='neg_mean_squared_error')
cv_score = pred.mean () 
print (pred.mean ()) 

#RandomForestRegressor
regrr = RandomForestRegressor(max_depth=2, random_state=0)
regrr.fit(features, target)

kf =KFold(n_splits=5) 
pred = cross_val_score(regrr,features, target, cv=kf, scoring='neg_mean_squared_error')
cv_score = pred.mean () 
print (pred.mean ()) 



#neural_network

regr = MLPRegressor(random_state=1, max_iter=500, activation='logistic',solver='lbfgs',hidden_layer_sizes=(50,15))
regr.fit(features, target)
regr.score(features, target)

kfo =KFold(n_splits=5) 
pred = cross_val_score(regr,features, target, cv=kfo, scoring='neg_mean_squared_error')
cv_score = pred.mean () 
print (pred.mean ())

kfo2 =KFold(n_splits=10) 
pred = cross_val_score(regr,features, target, cv=kfo2, scoring='neg_mean_squared_error')
cv_score = pred.mean () 
print (pred.mean ())




