import pandas as pd
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

print(df.head())



from sklearn.linear_model import LinearRegression
import pickle

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
