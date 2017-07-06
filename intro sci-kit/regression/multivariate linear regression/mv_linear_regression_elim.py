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

# Use Backwards Elimination to find a matrix of optimal features
import statsmodels.formula.api as sm
X_matrix = np.append(arr=np.ones(shape=(50,1)).astype(int), values=X_matrix, axis=1)

''' Multivariate Linear Regression '''
# X_opt is most simply a "copy" of X_matrix so that we don't completely lose old features
X_opt = X_matrix[:, [0, 1, 2, 3, 4, 5]]
# OLS algorithm is LinearRegression itself, we just have to fit it, now
regressor_OLS = sm.OLS(endog=Y_matrix, exog=X_opt).fit()

''' Now, we need to get important statistical information to figure out the 
    P-value of a feature and then ultimately remove it and rebuild the model'''
    
print(regressor_OLS.summary())

X_opt = X_matrix[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y_matrix, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X_matrix[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y_matrix, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X_matrix[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=Y_matrix, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X_matrix[:, [0, 3]]
regressor_OLS = sm.OLS(endog=Y_matrix, exog=X_opt).fit()

print(regressor_OLS.summary())

# In Review: Use sm.OLS(dep, indep).fit().predict(X_test)
# Used whole dataset X_matrix for predictions, nothing to test on.
# In future, use train_test_split and sm.OLS(train) and then .predict(test)
# Use Matplotlib to graph results