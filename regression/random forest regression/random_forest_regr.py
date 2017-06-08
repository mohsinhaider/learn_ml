# Implementation of Random Forest Regression
''' A regression algorithm that chooses "k" random training examples
    (x^i, y^i), builds a decision tree (using splits) from those chosen
    training examples, and then repeats the process of building the
    decision trees until N (desired) are made. A prediction scales across
    every tree, with the result being average of all of the outputs from
    each decision tree.'''
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X_matrix = dataset.iloc[:, [1]].values # Should be 2D
Y_matrix = dataset.iloc[:, 2].values # Should be 1D (vector)

# No missing, categorical, or enough to TTS data

''' Random Forest Regression -- a form of ensemble learning '''
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(X_matrix, Y_matrix)

prediction = rf_regressor.predict(np.array([[6.5]]))
print("The salary we predict for position level {} is {}".format(6.5, prediction[0]))

''' Non-continuity in the graph is due to visualization step not accounting 
    for intermediary values that end up showing us the straight line. '''
X_highres = np.arange(min(X_matrix), max(X_matrix), step=0.01)
X_highres = X_highres.reshape((X_highres.size, 1))
plt.scatter(X_matrix, Y_matrix, color="red")
plt.plot(X_highres, rf_regressor.predict(X_highres), color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary vs Position")
plt.show()

