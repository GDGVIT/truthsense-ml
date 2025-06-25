import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.base import clone
import joblib
import os

# ================================
# Load full original data
# ================================
X = np.load("outputs/X_balanced.npy")
y = np.load("outputs/y_balanced.npy")

# ================================
# Set up K-Fold strategy
# ================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ================================
# Base models (tuned versions)
# ================================
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

base_models = {
    "lightgbm": LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=100),
    "xgboost": XGBClassifier(objective="multi:softmax", num_class=3, eval_metric='mlogloss',
                             learning_rate=0.1, max_depth=5, n_estimators=100, use_label_encoder=False),
}

# ================================
# Create out-of-fold predictions
# ================================
meta_features = {name: np.zeros_like(y, dtype=int) for name in base_models}

for model_name, model in base_models.items():
    print(f"\nGenerating out-of-fold predictions for {model_name}...")
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y[train_idx]

        clf = clone(model)
        clf.fit(X_train_fold, y_train_fold)
        preds = clf.predict(X_val_fold)
        meta_features[model_name][val_idx] = preds  # assign OOF preds in correct places

# ================================
# Stack meta-features
# ================================
X_meta = np.column_stack([meta_features[name] for name in base_models])
y_meta = y  # Same targets

# ================================
# Final train/test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)

# ================================
# Train meta-model (Random Forest)
# ================================
print("\nTraining Meta Random Forest...")
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
meta_model.fit(X_train, y_train)

# ================================
# Evaluate meta-model
# ================================
y_pred = meta_model.predict(X_test)
print("\nClassification Report for Final Meta Random Forest (Leakage-Free):\n")
print(classification_report(y_test, y_pred))

# ================================
# Save model
# ================================
os.makedirs("model_weights_fluency", exist_ok=True)
joblib.dump(meta_model, "model_weights_fluency/random_forest_model.pkl")