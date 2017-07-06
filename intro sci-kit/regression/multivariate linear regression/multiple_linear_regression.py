''' Multivariate Linear Regression using all features (all-in approach) '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")

# Step 1: Seperate X (input) and Y (target)
X_matrix = dataset.iloc[:, :-1].values
Y_matrix = dataset.iloc[:, [-1]].values

# Step 2: Replace Missing Data (None)

# Step 3: Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X_matrix[:, -1] = encoder.fit_transform(X_matrix[:, -1])

hot_encoder = OneHotEncoder(categorical_features=[3])
X_matrix = hot_encoder.fit_transform(X_matrix).toarray()

# Multiple Colinearity: select n-1 dummy variables
X_matrix = X_matrix[:, 1:]

# Step 4: train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, Y_matrix, \
                                                    test_size=0.2, \
                                                    random_state=0)

''' Multivariate Linear Regression w/ LinearRegression '''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# As of now, this is an "All-In" approach -- we use all features
regressor.fit(X_train, Y_train)

# Predicting test set results
Y_test_predictions = regressor.predict(X_test)

# Would need countour plot for multi-dimensional plotting