# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:18:02 2021

@author: XHK

This code instantiates a 7 layer (not including initial and final layers) neural network, which was used to 
try to fit the vote-percentage-generating function. 

It takes data from the datavisualisation.py module or from files pickled by said module. 

It saves the neural net and the scaler used (standard-scaler) to pre-process the data. 
This is so another module (bestInputFinder.py) can access the neural-net model and scaler, and thus predict the succes
(measured in RankPercentile) of a randomly generated input. 

Coordination between the three modules  datavisualisation.py -> neuralnet.py -> bestInputFinder.py was all done
by hand. So not very automated. 



"""

import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import pandas as pd
import numpy as np 

## Keras
from keras.models import Sequential
from keras.layers import Dense

import os
cwd = os.getcwd()




new_data = True #if new data available: set to True
if new_data:
    print('Getting newest data')
    import datavisualisation #actually loads and cleans data
    
    data_init, data_other, data_total = datavisualisation.get_data()
    
data_init = pd.read_pickle("./data_init.pkl")
data_other = pd.read_pickle("./data_other.pkl")
data_total = pd.read_pickle("./data_total.pkl")
absent = pd.read_pickle("./absentees.pkl")
absent = absent[absent['RankPct']<0.60]    #use municipalities where we do poor and that have high abst
absentees = data_total[data_total['Municipality'].isin(absent['Municipality'].values)] #all data for high-abs municipalities
print('data_init has columns: ', list(data_init.columns.values))
print('data_other has columns: ', list(data_other.columns.values))
print('data_total has columns: ', list(data_total.columns.values))

X_vars_used = ['A1','A2','A3','D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']

#first just use data for only our party:
    

initial_p13 = pd.DataFrame()

def rank_init_data(y = data_init):
    y.index = range(len(y)) #removes duplicate_indexes
    y['Rank'] = y.groupby(by = 'Municipality')['Results'].rank(method = 'min', pct = True)
    
 
    return y 


data_init = rank_init_data(y = data_init)
use_only_party13 = False
X = pd.DataFrame()

if use_only_party13:
    
    initial_p13 = data_init[data_init['Party'] == 13]
    
    X = initial_p13[X_vars_used]
    X = X.append(data_other[X_vars_used])
    
    y= initial_p13['Results']
    y = y.append(data_other['Results'])

else:
 
    X = data_total[X_vars_used]
    y= data_total['RankPct']

abstainers = False
if abstainers: 
    print("Using ABSENTEE data: ")
    X = absentees[X_vars_used].dropna()
    y = 1- absentees['Abstention'].dropna()  #turn-out, to be maximized
    
use_only_initial_campaigns = False
if use_only_initial_campaigns:
     X = data_init[X_vars_used]
     
     
     #y= data_init['Results']
     y = data_init['Rank']
     y = data_init['Abstention']

#y = y/100     #so all numbers between 0 and 1 

X = X.values 


from keras.utils import to_categorical

 
#X[:,0:3] = X[:,0:3]/np.linalg.norm(X[:,0:3], axis = 0)   #normalizes the A1, A2, A3 allocations

print('X shape: ', X.shape)
y= y.values

    
print('Y shape: ', y.shape)




#BLOW UP Y
#y = y**2


import pickle as pkl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)
 

#mm_scaler = preprocessing.MinMaxScaler()
#X = mm_scaler.fit_transform(X)
 
#X = preprocessing.power_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

#y =  (y-np.mean(y))/np.std(y)   #normalizing....



###building the neural network: 
    
# define the keras model

num_inputs = len(X_vars_used)

model = Sequential()
model.add(Dense(14, input_dim=num_inputs, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(10,activation = 'relu'))   

#output layer: 
#model.add(Dense(1, activation='linear'))
model.add(Dense(1, activation='linear'))
# compile the keras model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X, y, epochs=2500, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

y_pred = model.predict(X_test)


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

model.save(os.getcwd()+r'\\Model26Jan2K')
 

print('Y-pred = ')
print(y_pred)
pkl.dump(scaler, open("Scaler26jan2k.pkl",'wb'))

abs_set = 0
if abstainers:
    abs_set = set(list(absentees.dropna()['Municipality'].values))
    pkl.dump(abs_set, open("abs_set.pkl",'wb'))
    
    
