import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

####### DATA PREPROCESSING

# load dataset as DataFrame
dataset = pd.read_csv('admissions_data.csv')
dataset = dataset.drop(['Serial No.'], axis=1)

# split data into features and labels parameters
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1] # column to predict

# split features and labels each into training and test sets
X_train, X_tests, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=42)


# standardize features so that they have equal weights
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.fit(X_tests)


# define neural network model
def design_model(features_set):
    learning_rate = 0.01
    model = Sequential()
    input = layers.InputLayer(input_shape=features_set.shape[1])
    # input layer
    model.add(input)
    # hidden layers
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    # output layer
    model.add(layers.Dense(1))
    opt = Adam(learning_rate)
    # compile model
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model
