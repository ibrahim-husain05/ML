import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("titanic.csv")
# print(df.head())
inputs= df.drop(["PassengerId", "Name", "Ticket", "SibSp","Cabin", "Parch", "Embarked", "Survived"], axis="columns")
target = df.Survived
# print(inputs.head())

from sklearn.preprocessing import LabelEncoder
Sex_le = LabelEncoder()
inputs["Sex_num"] = Sex_le.fit_transform(inputs.Sex)
# print(inputs.head())
inputs_n = inputs.drop(["Sex"], axis="columns")
# print(inputs_n.head())
model = DecisionTreeClassifier()
model.fit(inputs_n, target)
# print(model.score(inputs_n, target))
predictions = model.predict([[3,22.0,7.2500,1]])
print(predictions)