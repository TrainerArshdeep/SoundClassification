"""
Variables:

x : Storing data of features
y : Storing data of target
columns : To store the column names
lb : Label Encoder object used for encoding target variable
scaler : Min Max Scaler object used for scaling features
xtrain : To store training data of features
xtest : To store testing data of features
ytrain : To store training data of target
ytest : To store testing data of target

"""

# Importing relevant libraries

import pandas as pd # For dataframe manipulations
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For data visualization
import numpy as np # For array manipulations
from sklearn.preprocessing import LabelEncoder # For encoding target data
from sklearn.preprocessing import MinMaxScaler # For scaling the features
from sklearn.model_selection import train_test_split # To split the data into training and testing

# Reading features file
x=pd.read_csv('features.csv')

# Reading target file
y=pd.read_csv('target.csv')

# Dropping unnecesary columns
x.drop('Sr No',axis=1,inplace=True) 
y.drop('Unnamed: 0',axis=1,inplace=True)
  
lb=LabelEncoder() # Creating label encoder object
columns = y.columns # Storing column names of y
y = lb.fit_transform(y) # Encoding target variable
y =pd.DataFrame(y,columns=columns) # Converting the result into a DataFrame

scaler=MinMaxScaler() # Creating scaler object
columns = x.columns # Storing column names of features
x=scaler.fit_transform(x) # Scaling the features
x=pd.DataFrame(x,columns=columns) # Converting the result into a DataFrame

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=4) # Splitting the data into training and testing

xtrain.to_csv('xtrain.csv') # Storing the xtrain data as a csv file
xtest.to_csv('xtest.csv') # Storing the xtest data as a csv file
ytrain.to_csv('ytrain.csv') # Storing the ytrain data as a csv file
ytest.to_csv('ytest.csv') # Storing the ytest data as a csv file