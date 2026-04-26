# <p align="center">🌸 APEX IRIS FRAMEWORK 🌸</p>
## <p align="center">An Advanced Model Evaluation & Robustness Infrastructure</p>

<p align="center">
  <br>
  <a href="https://satyamshelke2005.github.io/Apex-Iris-Framework/">
    <img src="https://img.shields.io/badge/🚀%20ACCESS%20LIVE%20LEADERBOARD-VIEW%20RANKINGS-blueviolet?style=for-the-badge&logo=rocket&logoColor=white" width="700" />
  </a>
  <br>
  <br>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/System_Status-Active-00FFFF?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Platform-Independent-008080?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Security-RSA_Encrypted-00FFFF?style=for-the-badge" />
</p>

---

## 👥 Core Architecture Leads
| Name | PRN | Responsibility |
| :--- | :--- | :--- |
| **Satyam Anilrao Shelke** | 1132231165 | Lead Developer & DevOps Automation |
| **Harsha Purohit** | 1132231017 | Research & Model Robustness |
.............

---

## 🎯 Project Overview
The **Apex Iris Framework** is a high-performance system designed to stress-test classification models. It evaluates predictive integrity across clean datasets and corrupted feature sets simulated via distribution shifts and Gaussian noise.

### 🛠️ Innovation: Dynamic Percentile Thresholding
Standard models often suffer from **"Class Collapse"** when faced with high entropy or noise, defaulting to majority-class bias. Our framework implements a proprietary fix:
- **The Problem:** Static 0.5 decision boundaries fail under realistic feature corruption.
- **The Solution:** We utilize **Dynamic Percentile Thresholding**, calculating decision boundaries based on the top 35%-40% of softmax probabilities.
- **Result:** Maintains high precision and label diversity even in "noisy" environments.

---

## 📊 Dataset & Perturbation Profile
Our framework utilizes a specialized binary-mapped version of the **Iris Botanical Dataset**, optimized for Graph Topology Ablation testing.

### 🔍 Feature Schema
| Feature | Description | Unit |
| :--- | :--- | :--- |
| **Sepal Length** | Longitudinal measurement of the sepal | cm |
| **Sepal Width** | Lateral measurement of the sepal | cm |
| **Petal Length** | Longitudinal measurement of the petal | cm |
| **Petal Width** | Lateral measurement of the petal | cm |

### ⚠️ The Ablation Challenge
Models are evaluated on two distinct test pipelines:
1. **Standard Set:** Clean feature vectors representing the "Ideal" topology.
2. **Perturbed Set:** Features injected with **Gaussian Noise** ($\mu=0, \sigma=0.1$) and **Covariate Shifts** to simulate real-world sensor degradation.

---

## 📈 Technical Benchmarks
| Metric | Specification |
| :--- | :--- |
| **Architecture** | 3-Layer MLP with BatchNorm |
| **Optimization** | Adam ($lr=0.005$) |
| **Best Accuracy** | **98.2%** |
| **Loss Function** | Negative Log-Likelihood (NLL) |

---

## 📥 Submission Protocol
Follow these **6 steps** to integrate your model into the Apex evaluation pipeline:

### 1️⃣ Prepare Environment
```powershell
git clone [https://github.com/SatyamShelke2005/Apex-Iris-Framework.git](https://github.com/SatyamShelke2005/Apex-Iris-Framework.git)
cd Apex-Iris-Framework
pip install -r requirements.txt

```
### 2️⃣ Develop Your Model
Use the provided `data/train.csv` to train your classification model.

*  **Goal:** Aim for high accuracy while maintaining robustness against noise.
*  **Baseline:** Check `starter_code/baseline.py` for a simple starting point.


### 3️⃣ Generate Predictions
Run your trained model on the `data/test.csv` file. 

*  **Format:** Save your results as `predictions.csv`.
*  **Requirement:** Ensure your CSV contains the exact same number of rows as the test set.


### 4️⃣ Create Your Identity
Create a unique folder inside the `submissions/` directory to host your work:

- **Path:** `submissions/Your_Name_PRN/`
- Move your generated `predictions.csv` into this new folder.


### 5️⃣ Secure & Encrypt
Before pushing, you **must** encrypt your predictions using our RSA public key. This ensures your data remains private until the automated evaluation:
```powershell
python encryption/encrypt.py --input submissions/Your_Name_PRN/predictions.csv --key encryption/public_key.pem

```
### 6️⃣ Deploy & Rank
Commit your changes and open a Pull Request (PR) to the main repository:
```powershell
git add .
git commit -m "Official Submission: [Your Name]"
git push origin main
