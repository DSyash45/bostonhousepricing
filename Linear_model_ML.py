# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:05:45 2023

@author: 91966
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets

boston = datasets.load_boston()
type(boston)
boston.keys()

#check description of datasets 
print(boston.DESCR)

print(boston.data)
print(boston.feature_names)

#preparing datasets
datasets = pd.DataFrame(boston.data,columns =boston.feature_names)
datasets.head()

datasets['Price'] = boston.target

datasets.info()
datasets.describe()

#check miss value
datasets.isnull().sum()

#Exploratory data analysis
#correlation
datasets.corr()

import seaborn as sns
sns.pairplot(datasets)

plt.scatter(datasets['CRIM'],datasets['Price'])


#feature define

X = datasets.iloc[:,:-1]
Y = datasets.iloc[:,-1]

X.head()
Y.head()

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y,random_state=10,test_size=0.30)

#standardize the datasets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(train_x)
scaler.transform(test_x)

#Model traning
from sklearn.linear_model import LinearRegression
regression = LinearRegression().fit(train_x,train_y)

#print coeficient and intersept
print(regression.coef_)
print(regression.intercept_)

#prediction with test data
reg_pred = regression.predict(test_x);reg_pred

#scatter plot for prediction
plt.scatter(test_y,reg_pred)

#Residuals
residuals = test_y - reg_pred
residuals

#plotting residuals
sns.displot(residuals,kind = "kde")

#scatter plot between prediction & residuals.
plt.scatter(reg_pred,residuals) 

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
print(mean_squared_error(test_y,reg_pred))
print(mean_absolute_error(test_y,reg_pred))
print(np.sqrt(mean_absolute_error(test_y,reg_pred)))


#calculating R square
from sklearn.metrics import r2_score
score = r2_score(test_y,reg_pred)
print(score)

#accuracy score
regression.score(train_x,train_y)

#New Data Prediction
boston.data[0].shape
#reshape
boston.data[0].reshape(1,-1).shape

regression.predict(boston.data[0].reshape(1,-1))
# array([30.26475072])

#trasformation of new data
scaler.transform(boston.data[0].reshape(1,-1))

#prediction 
regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))
# array([39.44328942])

###########################..
# Picking The Model File for Deployment
import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))
pickle_model = pickle.load(open('regmodel.pkl','rb'))
pickle_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))





















