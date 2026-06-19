from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
digits = load_digits()
X, y = digits.data, digits.target
print(digits.target_names)
df = pd.DataFrame(digits.data,digits.target)
# print(df.head())
# print(dir(digits))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# rbf_svm = svm.SVC(kernel='rbf')
# print(len(x_train), len(x_test))
# rbf_svm.fit(x_train, y_train)
# print(rbf_svm.score(x_test, y_test))
# predicted = rbf_svm.predict(x_test)
# using linear kernel
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(x_train, y_train)
print(linear_svm.score(x_test, y_test))
