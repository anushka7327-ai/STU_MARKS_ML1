import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv("student_marks_dataset.csv")

X = data[["Study_Hours"]]
y = data["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model Evaluation")
print("----------------")
print("MAE :", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2  :", round(r2_score(y_test, predictions), 2))

# User prediction
hours = float(input("\\nEnter study hours: "))
predicted_marks = model.predict([[hours]])[0]
predicted_marks = max(0, min(100, predicted_marks))
print(f"Predicted marks: {predicted_marks:.2f}")

# Graph
plt.scatter(data["Study_Hours"], data["Marks"], label="Actual data")
plt.plot(data["Study_Hours"], model.predict(X), label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.legend()
plt.show()
