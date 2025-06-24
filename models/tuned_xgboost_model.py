import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
# Load balanced data
X = np.load("outputs/X_balanced.npy")
y = np.load("outputs/y_balanced.npy")

print("Loaded balanced data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200]
}

# Setup model + GridSearchCV
model = GridSearchCV(
    xgb.XGBClassifier(objective="multi:softmax", num_class=3, use_label_encoder=False, eval_metric='mlogloss'),
    param_grid,
    cv=5,
    verbose=1
)

# Train
print("\nTraining Tuned XGBoost model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report for Tuned XGBoost:\n")
print(classification_report(y_test, y_pred))

# Save predictions and y_test
np.save("outputs/xgboost_preds.npy", y_pred)
np.save("outputs/y_test.npy", y_test)

# After model.fit(...)
joblib.dump(model, "model_weights_fluency/xgboost_model.pkl")