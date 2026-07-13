import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Training Data (House Size to Price)
# X represents House Size (in square feet)
X_train = np.array([600, 900, 1200, 1500, 1800, 2200, 2500]).reshape(-1, 1)

# Y represents House Price (in thousands of dollars, e.g., 150 = $150,000)
Y_train = np.array([150, 210, 270, 330, 390, 470, 530])

# 2. Initialize and Train the Machine Learning Model
model = LinearRegression()
model.fit(X_train, Y_train)

print("--- House Price Predictor Model Trained! ---")

# 3. Get input from the user
user_input = input("Enter house size in square feet (e.g., 1400): ")

try:
    # Convert input to a decimal number
    house_size = float(user_input)

    if house_size < 1:
        print("Value can't be less than 1")
        import sys; sys.exit(1)
    
    # Reshape for scikit-learn format
    new_size_value = [[house_size]]

    # 4. Predict and display the result
    predicted_price = model.predict(new_size_value)[0]
    
    # Multiply by 1,000 since our training data was in thousands
    final_price_usd = predicted_price * 1000
    
    print(f"\nPredicted price for a {house_size:.0f} sq ft house: ${final_price_usd:,.2f}")

except ValueError:
    print("Please enter a valid number for the house size.")
