import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("beer-servings.csv")

# Display first 5 rows
print(df.head())

# Remove missing values
df.dropna(inplace=True)

# Select features
X = df[['beer_servings', 'spirit_servings', 'wine_servings']]

# Select target
y = df['total_litres_of_pure_alcohol']

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Create model
model = LinearRegression()

# Train model
model.fit(X_scaled, y)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully")