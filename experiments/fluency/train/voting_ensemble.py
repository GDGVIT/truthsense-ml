import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import xgboost as xgb
import joblib
import os

# Load balanced data
# Will need to change path according to where you run the file from
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "..", "outputs")
X = np.load(os.path.join(outputs_dir, "X.npy"))
y = np.load(os.path.join(outputs_dir, "y.npy"))


print("Loaded balanced data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define tuned base models
lgb_model = lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric='mlogloss',
                              learning_rate=0.1, max_depth=5, n_estimators=100, use_label_encoder=False)

# Voting ensemble
ensemble = VotingClassifier(estimators=[
    ('lgb', lgb_model),
    ('xgb', xgb_model)
], voting='hard')

# Train
print("\nTraining Voting Ensemble...")
ensemble.fit(X_train, y_train)

# Predict & Evaluate
y_pred = ensemble.predict(X_test)
print("\nClassification Report for Voting Ensemble (Tuned):\n")
print(classification_report(y_test, y_pred))

# Save
np.save("fluency/outputs/voting_preds.npy", y_pred)
np.save("fluency/outputs/y_test.npy", y_test)

# After voting_clf.fit(...)
joblib.dump(ensemble, "../weights/voting_model.pkl")