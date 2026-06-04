import pandas as pd
import sklearn.linear_model as lm
import numpy as np

# Load the data
df=pd.read_csv("homeprices.csv")
# print(df.head())

df.bedrooms = df.bedrooms.fillna(np.floor(df.bedrooms.median()))
print(df.bedrooms)
# Create the linear regression model
model=lm.LinearRegression()
# Fit the model 
model.fit(df[["area","bedrooms","age"]],df.price)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
# Predict the price f house with ar=3000, bedrooms=3, age=40
predicted_price=model.predict([[3000,3,40]])
print("Predicted price:", predicted_price[0])