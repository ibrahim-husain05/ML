import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
# Load 
data = pd.read_csv("canada_per_capita_income.csv")
print(data.head())
# Create the linear 
model = lm.LinearRegression()
# Fit the model to the data
model.fit(data[["year"]], data[["per capita income (US$)"]])
print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# Predict the per capita income for the year 2020
predicted_income = model.predict([[2020]])
print("Predicted per capita income for 2020:", predicted_income[0][0])
# Plot the data 
plt.scatter(data["year"], data["per capita income (US$)"], color="blue", label="Data")
plt.plot(data["year"], model.predict(data[["year"]]), color="red", label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Canada Per Capita Income (1978-2016)")
plt.legend()
plt.show()