import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
data = pd.read_csv("crop_yield_india.csv")

# Features & target
X = data[['Rainfall', 'Temperature', 'Area']]
y = data['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Save model
pickle.dump(model, open("crop_model.pkl", "wb"))

print("Model Saved Successfully")