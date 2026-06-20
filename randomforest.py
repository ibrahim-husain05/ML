from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
# print(iris.feature_names)
# print(iris.target_names)
df = pd.DataFrame(X, columns=iris.feature_names)
# print(df.head())
df['target'] = y
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)