import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

# Seperate X and Y (idp and dp)
X_matrix = dataset.iloc[:, [1]].values
Y_matrix = dataset.iloc[:, [-1]].values

# No missing data

# Categorical Data
#from sklearn.preprocessing import OneHotEncoder
#X_matrix = OneHotEncoder(categorical_features=[0]).fit_transform(X_matrix)\
#.toarray()
#print(X_matrix)


''' Linear Regression (for comparison) '''
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_matrix, Y_matrix)

''' Polynomial Regression
        1. PolynomialFeatures(degree=#).fit_transform() to modify matrix 
        2. LinearRegression().fit() with new matrix and original Y'''
from sklearn.preprocessing import PolynomialFeatures
polynomial_shift = PolynomialFeatures(degree=4)
X_polymod = polynomial_shift.fit_transform(X_matrix)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_polymod, Y_matrix)

''' LINEAR REGRESSION Y_matrix vs linear_regressor.predict(X_matrix) '''

plt.scatter(X_matrix, Y_matrix, color="blue")
plt.plot(X_matrix, linear_regressor.predict(X_matrix), color="red")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary vs Position (Linear Regression Fit)")
plt.show()

''' POLYNOMIAL REGRESSION Y_matrix vs linear_regressor_2.predict(X_matrix) '''

# Increase granularity of the curve by enlarging X_matrix
X_smaller_gran = np.arange(min(X_matrix), max(X_matrix), step=0.1)
X_smaller_gran = X_smaller_gran.reshape((len(X_smaller_gran), 1))

plt.scatter(X_matrix, Y_matrix, color="purple")
plt.plot(X_smaller_gran, linear_regressor_2.predict(polynomial_shift.fit_transform(\
                                                   X_smaller_gran)))
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary vs Position (Polynomial Regression Fit)")
plt.show()


''' PREDICT using Linear Regression '''
lin_salary_prediction = linear_regressor.predict(6.5)

''' PREDICT using Polynomial Regression '''
poly_salary_prediction = linear_regressor_2.predict(polynomial_shift.fit_transform(\
                                                    6.5))

print(lin_salary_prediction)
print(poly_salary_prediction)



