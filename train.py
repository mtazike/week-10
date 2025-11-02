import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)
print(df.head())

# Exercise 1: Linear Regression
X = df[['100g_USD']]
y = df['rating']

model = LinearRegression()
model.fit(X, y)

with open('model_1.pickle', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved as model_1.pickle")

# Exercise 2: Decision Tree Regressor
roast_map = {
    'Light': 0,
    'Medium-Light': 1,
    'Medium': 2,
    'Medium-Dark': 3,
    'Dark': 4
}
df['roast_num'] = df['roast'].map(roast_map)
print(df[['roast', 'roast_num']].head(5))

X = df[['100g_USD', 'roast_num']]
y = df['rating']

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X, y)

with open('model_2.pickle', 'wb') as f:
    pickle.dump(tree_model, f)
print("Decision Tree model trained and saved as model_2.pickle")

# Test prediction
sample_value = [[5, 2]]  # Example: price = 5 USD, roast = Medium
predicted_rating = tree_model.predict(sample_value)
print(f"Predicted rating for 100g_USD = 5 and roast_num = 2 is: {predicted_rating[0]:.2f}")
