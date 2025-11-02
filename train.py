import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

print(df.head())

# Exercise 1

# Select the feature (X) and target (y)
X = df[['100g_USD']]   # independent variable
y = df['rating']       # dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model as a pickle file
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model_1.pickle")

# Load from the pickle file
with open('model_1.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

# Test the model 
sample_value = [[5]]
predicted_rating = loaded_model.predict(sample_value)

print(f"Predicted rating for 100g_USD = 5 is: {predicted_rating[0]:.2f}")


# Exercise 2

# Create a mapping dictionary
roast_map = {
    'Light': 0,
    'Medium-Light': 1,
    'Medium': 2,
    'Medium-Dark': 3,
    'Dark': 4
}

# Apply mapping
df['roast_num'] = df['roast'].map(roast_map)

print(df[['roast', 'roast_num']].head(5))


# Select features (X) and target (y)
X = df[['100g_USD', 'roast_num']]   # two input features
y = df['rating']                    # target variable

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Create and train the Decision Tree model
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X, y)

# Save this model as model_2.pickle
with open('model_2.pickle', 'wb') as f:
    pickle.dump(tree_model, f)

print("Decision Tree model trained and saved as model_2.pickle")


# Load the saved model
with open('model_2.pickle', 'rb') as f:
    loaded_tree_model = pickle.load(f)

# Create a test sample
sample_value = [[5, 2]]   # example: price = 5 USD, roast = 'Medium'

# Predict rating
predicted_rating = loaded_tree_model.predict(sample_value)

print(f"Predicted rating for 100g_USD = 5 and roast_num = 2 is: {predicted_rating[0]:.2f}")

