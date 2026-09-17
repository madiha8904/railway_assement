import joblib
import pandas as pd
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest.joblib"
model = joblib.load(MODEL_PATH)

# New asset
new_asset = pd.DataFrame([{
    "asset_age": 14,
    "previous_failures": 3,
    "days_since_maintenance": 160,
    "usage": 85,
    "condition": 40,
    "criticality": 5
}])

# Predict
prediction = model.predict(new_asset)[0]

# Probability
probability = model.predict_proba(new_asset)[0][1]

if prediction == 1:
    priority = "HIGH"
else:
    priority = "LOW"

print("Predicted Priority:", priority)
print("Risk Probability:", round(probability, 2))
