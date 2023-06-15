import pandas as pd
from sklearn.model_selection import train_test_split

####### DATA PREPROCESSING

# load dataset as DataFrame
dataset = pd.read_csv('admissions_data.csv')
dataset = dataset.drop(['Serial No.'], axis=1)

# split data into features and labels parameters
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1] # column to predict

# split features and labels each into training and test sets
X_train, X_tests, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=42)

print(y_test)