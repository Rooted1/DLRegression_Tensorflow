import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

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
X_test_scale = sc.transform(X_tests)

####### CREATE MODEL AND TRAIN

# define neural network model
def design_model(features_set):
    learning_rate = 0.01
    model = Sequential()
    input = layers.InputLayer(input_shape=features_set.shape[1])
    # input layer
    model.add(input)
    # hidden layers
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dropout(0.2))
    # output layer
    model.add(layers.Dense(1))
    opt = Adam(learning_rate)
    # compile model
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

# test fit model with epochs and batch_size
epochs = 100
batch_size = 5
val_split = 0.3

model = design_model(X_train_scale)
earl_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
history = model.fit(X_train_scale, y_train.to_numpy(), epochs=epochs, batch_size=batch_size, verbose=1, validation_split=val_split, callbacks=[earl_stop])
res_mse, res_mae = model.evaluate(X_test_scale, y_test.to_numpy(), verbose=0)

print("MSE, MAE: ", res_mse, res_mae)

# evaluate r-squared value to see how well-fit the features are
y_predict = model.predict(X_test_scale, verbose=0)
r2_score = r2_score(y_test, y_predict)
print("R-squared score: ", r2_score)

# plot train and validation mae over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper right')

# Plot loss and validation loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper right')
plt.show()
