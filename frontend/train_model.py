
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = pd.read_csv(
    r"C:\Users\DELL\Downloads\stroke_risk_prediction_dataset.csv"
)

# 2. Remove unnecessary columns
df = df.drop(columns=[
    'Patient_ID',
    'Stroke_Risk',
    'AI_Health_Recommendation',
    'Doctor_Consultation_Needed'
])

# 3. Separate target and input columns
X = df.drop('Stroke_Risk_Score', axis=1)
y = df['Stroke_Risk_Score'].copy()

# 4. Identify numerical and categorical columns
num_cols = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

cat_cols = X.select_dtypes(
    include=['object']
).columns.tolist()

# 5. Save missing value replacement values
numeric_medians = X[num_cols].median().to_dict()

categorical_modes = {
    col: X[col].mode()[0] for col in cat_cols
}

# 6. Handle missing values
X[num_cols] = X[num_cols].fillna(numeric_medians)

for col in cat_cols:
    X[col] = X[col].fillna(categorical_modes[col])

# 7. Encoding
X = pd.get_dummies(
    X,
    columns=cat_cols,
    drop_first=True,
    dtype=int
)

encoded_columns = X.columns.tolist()

# 8. Outlier treatment using IQR
clip_bounds = {}

for col in X.select_dtypes(include=['number']).columns:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clip_bounds[col] = (lower, upper)

    X[col] = X[col].clip(lower, upper)

# Apply the same target outlier treatment as your notebook
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1

y = y.clip(
    Q1 - 1.5 * IQR,
    Q3 + 1.5 * IQR
)

# 9. Feature selection
selector = SelectKBest(
    score_func=f_regression,
    k=25
)

X_selected = selector.fit_transform(X, y)

# 10. Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_selected)

# 11. Train Linear Regression
model = LinearRegression()

model.fit(X_scaled, y)

# 12. Save model and preprocessing information
artifact = {
    'model': model,
    'selector': selector,
    'scaler': scaler,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'numeric_medians': numeric_medians,
    'categorical_modes': categorical_modes,
    'encoded_columns': encoded_columns,
    'clip_bounds': clip_bounds,
    'target': 'Stroke_Risk_Score'
}

joblib.dump(artifact, 'stroke_model.pkl')

print("Model saved successfully!")
print("Selected features:", 25)
print("Total encoded features:", len(encoded_columns))