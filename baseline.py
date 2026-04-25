import os
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from cryptography.fernet import Fernet
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------
# 1. Dynamic Automated Path Logic
# ----------------------------
# Using a specific identity for the Automated Merge Test
submitter_raw = os.getenv('SUBMITTER_NAME', 'Automated_Merge_Tester')
clean_name = submitter_raw.replace(" ", "_").replace(".", "_")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submissions", clean_name)
DATA_JSON_PATH = os.path.join(SCRIPT_DIR, "docs", "data.json")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Advanced Data & Model Logic
# ----------------------------
iris = load_iris()
X, y = iris.data, (iris.target == 1).astype(int)
# Changed random_state to ensure fresh evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=707)

scaler = PowerTransformer(method='yeo-johnson')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

class AdvancedResidualMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        identity = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(identity)))
        x = x + identity 
        return F.log_softmax(self.out(x), dim=1)

model = AdvancedResidualMLP(input_dim=4, num_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

print(f"--- Running Automated Workflow for: {submitter_raw} ---")
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = F.nll_loss(out, y_train_t)
    loss.backward()
    optimizer.step()

# ----------------------------
# 3. Metrics Generation
# ----------------------------
model.eval()
with torch.no_grad():
    out = model(X_test_t)
    probs = torch.exp(out)[:, 1].cpu().numpy()
preds = (probs >= 0.5).astype(int)

accuracy_val = np.mean(preds == y_test) * 100
f1_val = f1_score(y_test, preds, average='weighted') * 100

print(f"\n--- Results for {submitter_raw} ---")
print(f"Accuracy: {accuracy_val:.2f}%")
print(f"F1-Score: {f1_val:.2f}")

df_sub = pd.DataFrame({"row_index": range(len(preds)), "target": preds})
temp_csv = os.path.join(SUBMISSION_DIR, "temp.csv")
df_sub.to_csv(temp_csv, index=False)

# ----------------------------
# 4. Encryption
# ----------------------------
key = Fernet.generate_key() 
cipher_suite = Fernet(key)
with open(temp_csv, 'rb') as f:
    raw_data = f.read()
encrypted_data = cipher_suite.encrypt(raw_data)
with open(os.path.join(SUBMISSION_DIR, "final_submissions.csv.enc"), 'wb') as f:
    f.write(encrypted_data)
os.remove(temp_csv)

# ----------------------------
# 5. Metadata
# ----------------------------
# Making display name unique for this specific test
display_name = f"Test_User_{submitter_raw}"

metadata = {
    "name": display_name,
    "PRN": "AUTOTEST_7007",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "Residual MLP v3",
    "status": "Success",
    "accuracy": f"{accuracy_val:.2f}%"
}

with open(os.path.join(SUBMISSION_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

# ----------------------------
# 6. Automated Leaderboard Sync
# ----------------------------
new_entry = {
    "Participant": display_name,
    "Architecture": "Auto-Residual-Skip-v3", # Unique name for the board
    "Accuracy": f"{accuracy_val:.1f}%",
    "F1-Score": f"{f1_val:.1f}",
    "Timestamp": datetime.now().strftime("%Y-%m-%d")
}

try:
    if os.path.exists(DATA_JSON_PATH):
        with open(DATA_JSON_PATH, 'r') as f:
            leaderboard_data = json.load(f)
    else:
        leaderboard_data = []

    leaderboard_data = [e for e in leaderboard_data if e.get("Participant") != display_name]
    leaderboard_data.append(new_entry)

    with open(DATA_JSON_PATH, 'w') as f:
        json.dump(leaderboard_data, f, indent=4)
    print(f"\nLeaderboard successfully synced for: {display_name}")

except Exception as e:
    print(f"\nLeaderboard sync failed: {e}")

print(f"\n--- WORKFLOW COMPLETE ---")
