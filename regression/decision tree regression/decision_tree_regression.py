import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X_matrix = dataset.iloc[:, [1]].values
Y_matrix = dataset.iloc[:, [2]].values

# No missing data, no categorical data, no t_t_s

''' Decision Tree Regression '''
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_matrix, Y_matrix)

salary_prediction = dt_regressor.predict(np.array([[6.5]]))
print(salary_prediction)

''' Decision Trees Regression require higher resolution.
    DTR is non-linear, and not continuous. More x values are
    needed for splits that actually make sense, and thus contain
    more points in a given LEAF. '''

X_res_boost = np.arange(min(X_matrix), max(X_matrix), step=0.01)
X_res_boost= X_res_boost.reshape((X_res_boost.size, 1))
plt.scatter(X_matrix, Y_matrix, color="red")
plt.plot(X_res_boost, dt_regressor.predict(X_res_boost), color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary vs Position")
plt.show()

