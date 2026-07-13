# Here we aren't following any programming rules
# Such as Fahrenheit = Celsius * 1.8 + 32
# We are just giving AI the examples and asking to figure the
# rule out itself.

import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Training Data (Celsius to Fahrenheit)
# This is like a Teacher (The Data)
# The computer learns through this data, rather than we writing a rule it follows.
# in this case, we want the computer to convert the celsius to fahrenheit
# instead of we giving the rule Fahrenheit = Celsius * 1.8 + 32
# we give these data-sets, and asks the computer to figure the rule itself.

# python3
# Python 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> Fahrenheit = Celsius * 1.8 + 32
# KeyboardInterrupt
# >>> def c_to_f(celsius):
# ...     return celsius * 1.8 + 32
# ...
# >>> c_to_f(25)
# 77.0
# >>> c_to_f(25)
# 77.0
# >>> c_to_f(100)
# 212.0
# >>>

# X_train contains the questions (Celsius temperatures).
X_train = np.array([-40, -10, 0, 8, 15, 22, 38]).reshape(-1, 1)

# Y_train contains the correct answers (Fahrenheit temperatures).
Y_train = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4])


# using .reshape is a way to format data in a way the AI requires
"""
[[-40]
 [-10]
 [  0]
 [  8]
 [ 15]
 [ 22]
 [ 38]]
"""
# print(X_train)

# 2. Initialize and Train the Machine Learning Model

# creates a blank AI "brain"
# Linear Regression is a type of AI used when the relationship between numbers forms a straight line on a graph.
# If you plot Celsius vs. Fahrenheit on a graph, it forms a perfect straight line.
model = LinearRegression()

# .fit is the training phase.
# The AI looks at your flashcards (X_train and Y_train) over and over again.
# It notices patterns, like: "When Celsius is 0, Fahrenheit is 32."
# By the end of this line, the AI has successfully guessed the math formula connecting the two temperatures.
model.fit(X_train, Y_train)

# 3. Get input from the user
user_input = input("Enter a temperature in Celsius to convert: ")

try:
    # Convert input to a decimal number
    celsius_float = float(user_input)
    
    # Reshape the single value into a 2D array format required by scikit-learn
    new_celsius_value = [[celsius_float]]

    # 4. Predict and display the result
    # The .predict() function is the final exam.
    # You give the AI a brand new Celsius number it has never seen before.
    # Using the math formula it figured out during step 3, it instantly calculates and outputs the Fahrenheit answer.
    predicted_fahrenheit = model.predict(new_celsius_value)
    print(f"Predicted Fahrenheit for {celsius_float}°C is {predicted_fahrenheit[0]:.2f}.")

except ValueError:
    print("Please enter a valid number.")
