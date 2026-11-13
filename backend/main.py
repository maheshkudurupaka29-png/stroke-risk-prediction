
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

# Step 1: Load saved ML model
MODEL_PATH = Path(__file__).resolve().parent.parent / "stroke_model.pkl"

artifact = joblib.load(MODEL_PATH)

model = artifact['model']
selector = artifact['selector']
scaler = artifact['scaler']

num_cols = artifact['num_cols']
cat_cols = artifact['cat_cols']

numeric_medians = artifact['numeric_medians']
categorical_modes = artifact['categorical_modes']

encoded_columns = artifact['encoded_columns']
clip_bounds = artifact['clip_bounds']

# Step 2: Create FastAPI application
app = FastAPI()

# Step 3: Define input format
class PredictionInput(BaseModel):
    data: dict[str, Any]

# Step 4: Home route
@app.get("/")
def home():
    return {"message": "Stroke Prediction API is running"}

# Step 5: Health check
@app.get("/health")
def health():
    return {"status": "API is working"}

# Step 6: Get input feature names
@app.get("/schema")
def get_schema():
    return {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "numeric_defaults": numeric_medians,
        "categorical_defaults": categorical_modes
    }

# Step 7: Prediction route
@app.post("/predict")
def predict(input_data: PredictionInput):

    data = input_data.data

    required_columns = num_cols + cat_cols

    missing = [
        col for col in required_columns
        if col not in data
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={"missing_features": missing}
        )

    # Convert input into DataFrame
    new_data = pd.DataFrame([
        {col: data[col] for col in required_columns}
    ])

    # Handle numerical values
    for col in num_cols:

        new_data[col] = pd.to_numeric(
            new_data[col],
            errors='coerce'
        )

        new_data[col] = new_data[col].fillna(
            numeric_medians[col]
        )

    # Handle categorical values
    for col in cat_cols:

        new_data[col] = new_data[col].fillna(
            categorical_modes[col]
        ).astype(str)

    # One-hot encoding
    new_data = pd.get_dummies(
        new_data,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

    # Match training columns
    new_data = new_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Apply saved outlier limits
    for col, (lower, upper) in clip_bounds.items():

        new_data[col] = new_data[col].clip(
            lower=lower,
            upper=upper
        )

    # Feature selection
    selected_data = selector.transform(new_data)

    # Scaling
    scaled_data = scaler.transform(selected_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]

    return {
        "predicted_stroke_risk_score": round(
            float(prediction), 2
        )
    }
