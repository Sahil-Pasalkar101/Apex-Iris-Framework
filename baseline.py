import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. SETUP & DATA
# ----------------------------
# Use the environment variable for the PR folder name
submitter_name = os.getenv('SUBMITTER_NAME', 'Siya_Simple_Model')
SUBMISSION_DIR = f"submissions/{submitter_name.replace(' ', '_')}"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load Iris and binary-target (Versicolor)
X, y = load_iris(return_X_y=True)
y = (y == 1).astype(float).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling helps simple models converge faster
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 2. SIMPLE NUMPY MODEL
# ----------------------------
# Initialize weights (4 features, 1 output)
weights = np.zeros((4, 1))
bias = 0
lr = 0.1

# Training loop
for _ in range(100):
    # Predict (Forward)
    z = np.dot(X_train, weights) + bias
    preds = 1 / (1 + np.exp(-z))
    
    # Calculate error and update (Backward)
    error = preds - y_train
    weights -= lr * (np.dot(X_train.T, error) / len(y_train))
    bias -= lr * (np.sum(error) / len(y_train))

# ----------------------------
# 3. PREDICTIONS & CSV EXPORT
# ----------------------------
# Get final predictions on test set
test_z = np.dot(X_test, weights) + bias
final_probs = 1 / (1 + np.exp(-test_z))
final_preds = (final_probs >= 0.5).astype(int).flatten()

# Create the CSV file for the PR
output_path = os.path.join(SUBMISSION_DIR, "final_submissions.csv")
df_output = pd.DataFrame({
    "row_index": range(len(final_preds)),
    "target": final_preds
})

df_output.to_csv(output_path, index=False)

print(f"Model trained. Predictions saved to: {output_path}")
print("You can now stage, commit, and push this folder to create your PR.")