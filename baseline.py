import os
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
# 1. Dynamic Automated Path Logic & Ghost Filtering
# ----------------------------
# These are the names we want to BAN from the leaderboard forever
BLOCK_LIST = ["Satyam_Anilrao_Shelke", "SatyamShelke2005", "Test_User"]

submitter_raw = os.getenv('SUBMITTER_NAME', 'Satyam_Anilrao_Shelke')

# TERMINATE immediately if the current run is for a blocked user
if any(blocked_name in submitter_raw for blocked_name in BLOCK_LIST):
    print(f"!!! CRITICAL: Blocking execution for {submitter_raw} to prevent ghost folders !!!")
    exit(0)

clean_name = submitter_raw.replace(" ", "_").replace(".", "_")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submissions", clean_name)
DATA_JSON_PATH = os.path.join(SCRIPT_DIR, "docs", "data.json")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Data & Model Logic
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

print(f"--- Training Model for {submitter_raw} ---")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = F.nll_loss(out, y_train_t)
    loss.backward()
    optimizer.step()

# ----------------------------
# 3. Generating Predictions & Metrics
# ----------------------------
model.eval()
with torch.no_grad():
    out = model(X_test_t)
    probs = torch.exp(out)[:, 1].cpu().numpy()
preds = (probs >= 0.5).astype(int)

accuracy_val = np.mean(preds == y_test) * 100
f1_val = f1_score(y_test, preds, average='weighted') * 100

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

if os.path.exists(temp_csv):
    os.remove(temp_csv)

# ----------------------------
# 5. Metadata Generation (DYNAMIC VERSION)
# ----------------------------
# This displays the GitHub username of whoever pushes/merges
display_name = submitter_raw 

# Assign your PRN only to you (based on username variations)
if "Satyam" in submitter_raw or "Shelke" in submitter_raw:
    prn = "1132231165"
else:
    prn = "EXTERNAL_CONTRIBUTOR"

metadata = {
    "name": display_name,
    "PRN": prn,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "PyTorch RobustMLP",
    "status": "Success",
    "accuracy": f"{accuracy_val:.2f}%",
    "submission_type": "Automated_CI_CD"
}

with open(os.path.join(SUBMISSION_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

# ----------------------------
# 6. Automated Leaderboard Sync & Scrubbing
# ----------------------------
new_entry = {
    "Participant": display_name,
    "Architecture": "PyTorch RobustMLP",
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

    # SCRUBBING: Remove old "Satyam Shelke" entry and any existing entry for this user
    # to allow the new score to overwrite the old one.
    NAMES_TO_SCRUB = ["Satyam Anilrao Shelke", "Satyam Shelke", "Test_User"]
    
    leaderboard_data = [
        e for e in leaderboard_data 
        if e.get("Participant") not in NAMES_TO_SCRUB 
        and e.get("Participant") != display_name 
    ]
    
    # Append the new clean entry
    leaderboard_data.append(new_entry)

    with open(DATA_JSON_PATH, 'w') as f:
        json.dump(leaderboard_data, f, indent=4)
    print(f"\nLeaderboard successfully updated for: {display_name}")

except Exception as e:
    print(f"\nLeaderboard update failed: {e}")

print(f"\n--- PROCESS COMPLETE ---")

#Code end
