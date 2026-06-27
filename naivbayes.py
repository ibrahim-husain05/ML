
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split    
import pandas as pd
wine=load_wine()
df=pd.DataFrame(wine.data, columns=wine.feature_names)
# print(df.head())
X=wine.data
y=wine.target
# print(wine.feature_names)
# print(wine.target_names)
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# using Gaussian Naive Bayes model
Gaussian_model=GaussianNB()
Gaussian_model.fit(X_train, y_train)
print(Gaussian_model.score(X_test, y_test))
# using Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
Multinomial_model=MultinomialNB()
Multinomial_model.fit(X_train, y_train)
print(Multinomial_model.score(X_test, y_test))

