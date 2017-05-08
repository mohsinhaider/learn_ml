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

# Categorial Data splitting
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

country_encoder = LabelEncoder()
f_matr[:, 0] = country_encoder.fit_transform(f_matr[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
features_matrix = onehotencoder.fit_transform(features_matrix).toarray()

response_encoder = LabelEncoder()
r_matr = response_encoder.fit_transform(r_matr)

# Separating training and testing set
from sklearn.preprocessing import train_test_split
f_matr_train, f_matr_test,\
r_matr_train, r_matr_test = train_test_split(f_matr, r_matr, test_size=0.2, random_state=0)

# Feature Scaling -- age and salary columns should be same range
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
f_matr_train = standard_scaler.fit_transform(f_matr_train)
f_matr_test = standard_scaler.fit_transform(f_matr_test)
