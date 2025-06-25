import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import joblib
import os
os.makedirs("model_weights_fluency", exist_ok=True)
# Load balanced dataset
X = np.load("outputs/X_balanced.npy")
y = np.load("outputs/y_balanced.npy")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0]
}

# Grid Search
grid = GridSearchCV(LGBMClassifier(), params, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
print("\nBest Params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("\nClassification Report for Tuned LightGBM:\n")
print(classification_report(y_test, y_pred))

# After model.fit(...)
joblib.dump(grid, "model_weights_fluency/lightgbm_model.pkl")
print("Model saved.")