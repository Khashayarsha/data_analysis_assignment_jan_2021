# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:49:53 2021

@author: XHK



takes data from datavisualisation.py module and then runs a panel-regression on it, as defined in the report.
Saves the regression output to a .txt file named summary3.txt    (contains p-values and parameter estimates etc)





"""
################################ Loading Relevant Data ###################################
import numpy as np 
import pandas as pd 

# Display 6 columns for viewing purposes
pd.set_option('display.max_columns', 7)

# Reduce decimal points to 2
#pd.options.display.float_format = '{:,.2f}'.format

new_data = False #if new data available: set to True
if new_data:
    print('Getting newest data')
    import datavisualisation #actually loads and cleans data
    
    data_init, data_other, data_total = datavisualisation.get_data()

data_init = pd.read_pickle("./data_init.pkl")
data_other = pd.read_pickle("./data_other.pkl")
absent = pd.read_pickle("./absentees.pkl")



data = data_other #[data_other['Results']] #data_init.append(data_other)
cluster_dummies = pd.get_dummies(data['Cluster'])

c_names = ['C'+str(i) for i in range(15)]
data[c_names] = pd.get_dummies(data['Cluster'], drop_first = True)

data.index = range(len(data))
data['A1sq'] = data['A1'].apply(lambda x: x**2)
data['A2sq'] = data['A2'].apply(lambda x: x**2)
data['A1A2'] = data['A1']*data['A2']
data['A1sqA2sq'] = data['A1sq']*data['A2sq']
#data['ex'] = data['A1A2'].apply(lambda x: np.e**(1+x)).astype(float)

data1 = data 


data = data.reset_index().set_index(['Municipality','Day']) #multi index to compatibilize with 
#https://bashtage.github.io/linearmodels/doc/panel/pandas.html

embeddings = ['E0', 'E1', 'E2']
#############################################################################################
import statsmodels.api as sm
demographics = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13',  'D14', 'D15' ,'E0', 'E1', 'E2']
reg1 = [  'A1', 'A2',   'A1sq', 'A2sq','A1A2', 'A1sqA2sq']  #+demographics
reg1 = reg1 + c_names 

X  = data[reg1]


dependent = data.RankPct 
normalize = False
if normalize:
    
    X = (X-X.mean())/X.std()
    
    
    
    
    
    dependent = (dependent - dependent.mean()) / dependent.std()

log = True
if log and not normalize:   #otherwise might try to take logs of negative numbers
    dependent = dependent.apply(lambda x: np.log(x))
     
exog = sm.add_constant(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(exog,dependent,test_size = 0.1)

from linearmodels import PanelOLS

from linearmodels.panel import RandomEffects


#mod = RandomEffects(y_train, X_train )
mod = PanelOLS(y_train, X_train, entity_effects = True,   drop_absorbed = True)
res = mod.fit(cov_type='heteroskedastic')

mod2 = RandomEffects(y_train, X_train)
res2 = mod2.fit(cov_type='heteroskedastic')

print(res)
print(res2)
#mod2.predict(X_test)


from sklearn import linear_model
mod3 = linear_model.LinearRegression(fit_intercept = True)



municipality_dummies = pd.get_dummies(data1['Municipality'])
data1[municipality_dummies.columns] = municipality_dummies

y1 = data1.RankPct

log = True
if log:   #otherwise might try to take logs of negative numbers
    y1 = y1.apply(lambda x: np.log(x)).values

reg2 = [  'A1', 'A2',   'A1sq', 'A2sq','A1A2', 'A1sqA2sq'] +list(municipality_dummies.columns)

x1 = data1[reg2] 
#x1['B0'] = 1

X_train,X_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.1)


mod = sm.OLS(y_train, X_train)

res = mod.fit(cov_type = 'HC1')

print(res.summary())



with open('summary3.txt', 'w') as fh:
    fh.write(res.summary().as_text())

with open('summary.csv', 'w') as fh:
    fh.write(res.summary().as_csv())



from statsmodels.tools.eval_measures import rmse

y_pred = res.predict(X_test)

y_pred = np.e**y_pred
y_test = np.e**y_test

mse = rmse(y_pred, y_test)
print(mse)

from scipy.optimize import minimize
def predicted_rankpct(args, municipality):
    bmunicipality = res.params[municipality]
    print(bmunicipality)
    a1, a2 = args
    b1,b2,b1sq,b2sq,b1b2,b1sqb2sq = [4.9289, 6.0537, -4.84, -6.2052, -9.0249, 11.6577 ]

    result = b1*a1 + b2 * a2 + b1sq * a1**2 + b2sq * a2**2 + b1b2 * a1 * a2 + b1sqb2sq*(a1**2)*(a2**2) + bmunicipality
    return -result
                           
test_mun = data1['Municipality'].iloc[210]
x0 = [0.5,0.5]
result = res = minimize(predicted_rankpct, x0, method='nelder-mead',

               options={'xatol': 1e-8, 'disp': True}, args = (test_mun))

print(result)