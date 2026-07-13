import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Training Data (Celsius to Fahrenheit)
X_train = np.array([-40, -10, 0, 8, 15, 22, 38]).reshape(-1, 1)
Y_train = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4])

# 2. Initialize and Train the Machine Learning Model
model = LinearRegression()
model.fit(X_train, Y_train)

# 3. Get input from the user
user_input = input("Enter a temperature in Celsius to convert: ")

try:
    # Convert input to a decimal number
    celsius_float = float(user_input)
    
    # Reshape the single value into a 2D array format required by scikit-learn
    new_celsius_value = [[celsius_float]]

    # 4. Predict and display the result
    predicted_fahrenheit = model.predict(new_celsius_value)
    print(f"Predicted Fahrenheit for {celsius_float}°C is {predicted_fahrenheit[0]:.2f}.")

except ValueError:
    print("Please enter a valid number.")
