# Deep Learning Regression with Admission Data

## Description

This is a codecademy regression challenge project where I used the admissions data from [Kaggle graduate admissions dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv) to train a model that predicts the likelihood that a student applying to graduate school will be accepted based on the features of the dataset.

## Requirements

Completing this project include 

+ preprocessing the data and removing series that may affect the final predictions
+ splitting the data into features and labels and also making sure to map categorical variables to numerical if any
+ splitting `features` and `labels` each into `training` and `test` sets using `scikit-learn`
+ normalizing the features to have equal weights
+ creating the neural network by adding the input, hidden, and output layers, setting the activation and loss functions and metrics, setting the gradient descent optimizer as well as the learning rate
+ fitting the model and evaluating with the test set for accuracy
+ performing hyperparamter tuning to improve performance of training data
+ evaluating the rsquared score to see how well features data performed in predicting an applicant's admission into a graduate program
+ plotting graphs of results from each training

## Conclusion

The graphs from training the model and hyperparameter tuning are in the `.png` files.

+ MSE = 0.0044
+ MAE = 0.0471
+ R2 score = 0.7858