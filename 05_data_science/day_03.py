import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load data set
data = pd.read_csv("experience_salary.csv")

x = data[["YearsExperience"]]
y = data[["Salary"]]

model = LinearRegression()
model.fit(x, y)

data["PredictedSalary"] = model.predict(x)

print("Model Coefficient (slope)", round(float(model.coef_[0].item()), 2))
print("Model Intercept (base salary)", round(float(model.intercept_.item()), 2))

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, data["PredictedSalary"], color="red", label="Regression line")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()