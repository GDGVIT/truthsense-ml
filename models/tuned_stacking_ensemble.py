import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import xgboost as xgb
import joblib
import os

# Load data
X = np.load("outputs/X_balanced.npy")
y = np.load("outputs/y_balanced.npy")

print("Loaded balanced data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define base models
lgb_model = lgb.LGBMClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42
)
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax", num_class=3, eval_metric='mlogloss',
    learning_rate=0.1, max_depth=5, n_estimators=100,
    use_label_encoder=False, random_state=42
)

# Define meta model
meta_model = LogisticRegression(max_iter=1000)

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
    final_estimator=meta_model,
    passthrough=False,  # Only use base model outputs
    cv=5,
    n_jobs=-1
)

# Train
print("\nTraining Stacking Ensemble...")
stacking_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacking_clf.predict(X_test)
print("\nClassification Report for Stacking Ensemble (Tuned):\n")
print(classification_report(y_test, y_pred))

# Save predictions and model
np.save("outputs/stacking_preds.npy", y_pred)
joblib.dump(stacking_clf, "model_weights_fluency/stacking_model.pkl")