from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# predictions = model.predict([[5.1, 3.5, 1.4, 0.2] ])
# print(iris.target_names[predictions[0]])
predictions = model.predict([[6.7, 3.1, 4.7, 1.5]])
print(iris.target_names[predictions[0]])