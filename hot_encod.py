import pandas as pd
import sklearn.linear_model as lm
model=lm.LinearRegression()
# Load the data
df=pd.read_csv("carprices.csv")
# print(df.head())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=df
dfle["Car Model"]=le.fit_transform(dfle["Car Model"])
# print(dfle)
X=dfle[["Car Model","Mileage","Age(yrs)"]]
# print(X)
y=dfle["Sell Price($)"]
# print(y)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([("Car Model", OneHotEncoder(), [0])], remainder="passthrough")
X=ct.fit_transform(X)
# print(X)
X = X[:,1:]
# print(X)
model.fit(X,y)

# predicting for mercedez benz age=4yr and mileage=45000
prediction = model.predict([[0,1,45000,4]])
print('Predicted price:', prediction[0])


