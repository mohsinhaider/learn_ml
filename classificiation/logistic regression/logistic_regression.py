''' Logistic Regression -- Binary Classification Problem '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")

X_matrix = dataset.iloc[:, [2, 3]].values
Y_matrix = dataset.iloc[:, -1].values

# Creating the test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, Y_matrix,\
                                                    test_size=0.25,
                                                    random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

''' Logistic Regression '''
from sklearn.linear_model import LogisticRegression
logistic_regressor = LogisticRegression(random_state=0)
logistic_regressor.fit(X_train, Y_train)

Y_test_pred = logistic_regressor.predict(X_test)

# Make the confusion matrix
from sklearn.metrics import confusion_matrix
# add the first diagonal for correct predictaions, second for incorrect predictions
conf_matr = confusion_matrix(Y_test, Y_test_pred)

# Visualizing Logistic Regression results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.5, stop = X_set[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X1, X2, logistic_regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


