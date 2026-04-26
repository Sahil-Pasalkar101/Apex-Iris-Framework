import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cryptography.fernet import Fernet
from sklearn.metrics import f1_score

# ----------------------------
# 1. DIRECTORY TARGETING
# ----------------------------
submitter_raw = os.getenv('SUBMITTER_NAME', 'Satyam_Anilrao_Shelke')

# The SHIELD: Stops these names from creating folders in the main repo
BLOCK_LIST = ["SatyamShelke2005", "Satyam_Anilrao_Shelke", "Test_User"]

if any(name in submitter_raw for name in BLOCK_LIST):
    if os.getenv('GITHUB_ACTIONS'):
        print(f"DEBUG: Skipping automation for admin user: {submitter_raw}")
        sys.exit(0)
    else:
        # Local safety: prevents creating a folder if running in the root
        if "submissions" not in os.getcwd():
            print("!!! SAFEGUARD: Blocked local execution in ROOT.")
            sys.exit(0)

clean_name = submitter_raw.replace(" ", "_").replace(".", "_")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submissions", clean_name)
DATA_JSON_PATH = os.path.join(SCRIPT_DIR, "docs", "data.json")

# Force create the specific submitter's folder
os.makedirs(SUBMISSION_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. DATA & MODEL
# ----------------------------
iris = load_iris()
X, y = iris.data, (iris.target == 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

class RobustMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.out(x), dim=1)

model = RobustMLP(input_dim=4, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

# ----------------------------
# 3. GENERATE FILES (.enc and .json)
# ----------------------------
model.eval()
with torch.no_grad():
    out = model(X_test_t)
    probs = torch.exp(out)[:, 1].cpu().numpy()
preds = (probs >= 0.5).astype(int)

accuracy_val = np.mean(preds == y_test) * 100
f1_val = f1_score(y_test, preds, average='weighted') * 100

# Write Temp CSV
temp_csv = os.path.join(SUBMISSION_DIR, "temp.csv")
pd.DataFrame({"row_index": range(len(preds)), "target": preds}).to_csv(temp_csv, index=False)

# ENCRYPTION
raw_key = os.getenv('ENCRYPTION_KEY')
key = raw_key.encode() if raw_key else Fernet.generate_key()
cipher_suite = Fernet(key)

with open(temp_csv, 'rb') as f:
    encrypted_data = cipher_suite.encrypt(f.read())

# SAVING THE .ENC FILE
enc_path = os.path.join(SUBMISSION_DIR, "final_submissions.csv.enc")
with open(enc_path, 'wb') as f:
    f.write(encrypted_data)

if os.path.exists(temp_csv):
    os.remove(temp_csv)

# SAVING THE METADATA
prn = "1132231165" if "Satyam" in submitter_raw else "EXTERNAL_CONTRIBUTOR"
metadata = {
    "name": submitter_raw,
    "PRN": prn,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": f"{accuracy_val:.2f}%"
}

with open(os.path.join(SUBMISSION_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

# LEADERBOARD SYNC
new_entry = {"Participant": submitter_raw, "Architecture": "PyTorch RobustMLP", "Accuracy": f"{accuracy_val:.1f}%", "Timestamp": datetime.now().strftime("%Y-%m-%d")}
try:
    if os.path.exists(DATA_JSON_PATH):
        with open(DATA_JSON_PATH, 'r') as f: data = json.load(f)
    else: data = []
    data = [e for e in data if e.get("Participant") != submitter_raw]
    data.append(new_entry)
    with open(DATA_JSON_PATH, 'w') as f: json.dump(data, f, indent=4)
except Exception as e:
    print(f"Leaderboard sync failed: {e}")
