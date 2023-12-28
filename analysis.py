"""
Variables:

dff : Read and store feature data
dff1 : Store data in dff after changing a column name
dft : Read and store target data
dft1 : Store data in dft after changing a column name
dfm : Storing merged dataframe containing features and target
dfm2 : Storing the descriptive statistics of dfm

Functions:

createbarchart : Display a bar plot
createpiechart : Display a pie chart
createheatmap : Display a heatmap

"""

# Importing relevant libraries

import pandas as pd # To analyze data
import os # To generate file path
import seaborn as sns # For visualizations
import numpy as np # For array manipulation
import matplotlib.pyplot as plt # For visualizations

dff=pd.read_csv(r'features.csv',index_col=0) #reading features.csv file from file path
dff1=dff.rename(columns={dff.columns[4]:'4'}) #renaming unnamed columns

dft=pd.read_csv(r'target.csv',index_col=0) # reading target.csv file 
dft1=dft.rename(columns={dft.columns[0]: 'Sounds'}) # renaming the unnamed column

dfm=pd.concat([dft1,dff1],axis=1) # Joining two dataframes to get data from two different dataframes into one dataframe

# creating function of createbarchart for bar plot visualization
def createbarchart():
    sns.countplot(data=dfm ,y= 'Sounds')
    plt.savefig('static/barchart.png')

def createpiechart(): # creating function to plot pie chart 
    value_counts_result = dfm['Sounds'].value_counts()
    mylabels=dfm['Sounds'].unique()             
    plt.figure(figsize=(7, 7))
    plt.pie(value_counts_result,labels=mylabels)
    plt.title('Pie Chart for Sound')
    plt.savefig('static/piechart.png')

dfm2=dfm.describe() # Running describe() function to get descriptive analysis

def createheatmap(): # creating heatmap function 
    plt.figure(figsize=(15,10))
    sns.heatmap(dfm2,cmap='RdBu',linewidth=0.5)
    plt.savefig('static/heatmap.png')


createbarchart()
createpiechart()
createheatmap()