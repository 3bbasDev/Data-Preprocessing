import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as sp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

# Missing value with nan
impute = sp(missing_values=np.nan, strategy='mean')
H = impute.fit(X[:, 1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])
# print(X)

# Encode column to make easy to run with ml --> convert category to three column like iq qp ln ==> 100 010 001
ColumnTrans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ColumnTrans.fit_transform(X))
print(X)

# Encode column to make easy to run with ml --> convert category in one column like yse not ==> 0 1
y = LabelEncoder().fit_transform(y)
# print(y)

# Train and Test
# Splitting dbs to tran and test it
X_train, X_test, Y_Train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print(X_train)
print(X_test)
print(Y_Train)
print(Y_test)
