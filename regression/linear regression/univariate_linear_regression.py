import pandas as pd
import numpy as np

''' PREPROCESSING '''

# Import the datatset
dataset = pd.read_csv("Salary_Data.csv")

# Seperate features and responses
feature_matrix = dataset.iloc[:, [0]].values
response_matrix = dataset.iloc[:, [1]].values

# Split the training and testing data sets
from sklearn.cross_validation import train_test_split
feature_train, feature_test, reponse_train, response_test =\
    train_test_split(feature_matrix, response_matrix, \
                     test_size=1/3, random_state=0)
    
''' Fitting Linear Regression into Training Set (auto-featscale)'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(feature_train, resp_train)

''' Predicing Test Set values using LinearRegression.predict()'''
regressor.predict(feature_test)