import pandas as pd

####### DATA PREPROCESSING

# load dataset as DataFrame
dataset = pd.read_csv('admissions_data.csv')
dataset = dataset.drop(['Serial No.'], axis=1)

# split data into features and labels parameters
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1] # column to predict