import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Training Data (Celsius to Fahrenheit)
# X represents our inputs (Celsius)
X_train = np.array([-40, -10, 0, 8, 15, 22, 38]).reshape(-1, 1)

# Y represents our known outputs (Fahrenheit)
Y_train = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4])

# 2. Initialize and Train the Machine Learning Model
model = LinearRegression()
model.fit(X_train, Y_train)

# 3. Test the Model with a new number (e.g., 25 degrees Celsius)
new_celsius_value = [[30]]

predicted_fahrenheit = model.predict(new_celsius_value)

print(f"Predicted Fahrenheit for {new_celsius_value[0][0]}°C is {predicted_fahrenheit[0]:.2f}")