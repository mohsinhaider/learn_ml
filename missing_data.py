import pandas as pd

dataset = pd.read_csv("Data.csv")

# Seperate features and responses
f_matr = dataset.iloc[:, :-1].values
r_matr = dataset.iloc[:, -1].values

# Fill in missing data with mean method
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(f_matr[:, [1,2]])
f_matr[:, [1,2]] = imputer.transform(f_matr[:, [1,2]])