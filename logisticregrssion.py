import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("HR_comma_sep.csv")
left = df[df.left==1]
print(left.shape)
remaining = df[df.left==0]
print(remaining.shape)
print(df.groupby('left').mean(numeric_only=True))

# print(df.head())
# print(df.dtypes)
# pd.crosstab(df.salary,df.left).plot(kind="bar")
# plt.show()
# pd.crosstab(df.Department,df.left).plot(kind="bar")
# plt.show()
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(subdf.head())

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# One-hot encode the categorical salary feature using sklearn
categorical_features = ['salary']
encoder = ColumnTransformer(
    transformers=[
        ('salary', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

X = encoder.fit_transform(df[['salary', 'satisfaction_level','average_montly_hours','promotion_last_5years']])
y = df['left']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
model = LogisticRegression()
model.fit(X_train,y_train)  
print("Model score:", model.score(X_test,y_test))