# 🛡️ Credit Card Fraud Detection System
### A Machine Learning Project | Python · XGBoost · SMOTE · Streamlit

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project builds a complete **end-to-end fraud detection pipeline** on the Kaggle Credit Card Fraud dataset. It covers every stage of the machine learning lifecycle — from exploratory data analysis through model training, rigorous evaluation, and a deployed interactive web application.

The dataset contains **284,807 transactions** made by European cardholders in September 2013, of which only **492 (0.17%)** are fraudulent — a severe **578:1 class imbalance** that makes this a challenging and realistic ML problem.

---

## 🎯 Objective

> Identify fraudulent credit card transactions in real time using machine learning, while correctly handling extreme class imbalance.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | [Kaggle — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total rows | 284,807 transactions |
| Fraud cases | 492 (0.17%) |
| Legitimate | 284,315 (99.83%) |
| Features | V1–V28 (PCA-transformed) + Amount + Time |
| Target | `Class` — 0 = Legitimate, 1 = Fraud |
| Missing values | None |

> **Note:** Features V1–V28 are principal components obtained via PCA to protect user confidentiality. Amount and Time are the only original unscaled features.

---

## 🏗️ Project Pipeline

```
Raw Data (creditcard.csv)
        │
        ▼
┌──────────────────┐
│  Phase 1 — EDA   │  Class imbalance · Amount/Time distributions
│                  │  Feature correlations · Top fraud indicators
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 2 — Preprocess  │  StandardScaler (Amount & Time)
│                        │  80/20 stratified split
│                        │  SMOTE on training set only
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 3 — Training    │  Logistic Regression (baseline)
│                        │  Random Forest (bagging)
│                        │  XGBoost (boosting) ← best
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 4 — Evaluation  │  AUC-ROC · Recall · F1 · MCC
│                        │  Confusion matrices · ROC curves
│                        │  Threshold tuning · Cross-validation
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Phase 5 — Deployment  │  Streamlit 4-tab web dashboard
│                        │  Single predict · Batch CSV · Analytics
└────────────────────────┘
```

---

## 📈 Model Results

| Model | AUC-ROC | Recall | Precision | F1-Score | MCC | Fraud Caught |
|-------|---------|--------|-----------|----------|-----|-------------|
| Logistic Regression | 0.9700 | 0.9184 | 0.0581 | 0.1094 | 0.1060 | 90 / 98 |
| Random Forest | 0.9788 | 0.8776 | 0.2976 | 0.4444 | 0.4200 | 86 / 98 |
| XGBoost (default 0.50) | 0.9800 | 0.8776 | 0.2614 | 0.4028 | 0.4776 | 86 / 98 |
| **XGBoost (tuned 0.85)** | **0.9800** | **0.8469** | **0.6486** | **0.7345** | **0.7406** | **83 / 98** |

> ✅ **Selected model:** XGBoost with tuned threshold 0.85 — highest F1 (0.7345) and MCC (0.7406).
>
> ⚠️ **Why not accuracy?** A model predicting everything as legitimate scores 99.83% accuracy while catching zero fraud. **Recall** and **AUC-ROC** are the correct metrics.

---

## 🔑 Key Findings

- **Class imbalance is severe** — 578:1 ratio; plain accuracy is useless as a metric
- **SMOTE** balanced training data from 394 fraud cases to 227,451 synthetic samples (1:1 ratio)
- **V14 dominates** — contributes 52.4% of XGBoost's predictions; V14 < -5 = high fraud risk
- **Threshold tuning** improved F1 from 0.40 → 0.73 and reduced false alarms from 243 → 45
- **MCC** is the most reliable single metric when classes are severely imbalanced

---

## 🖥️ Streamlit Dashboard

The web app has 4 tabs:

| Tab | Features |
|-----|---------|
| 🔮 **Prediction** | Real-time fraud gauge chart, probability bar, risk tier badge (Low/Medium/High/FRAUD) |
| 📊 **Analytics** | Feature importance, model comparison bar chart, confusion matrix |
| 📂 **Batch Predict** | Upload CSV → predict all rows → download flagged transactions |
| ℹ️ **About** | Full pipeline summary and model explanation |

**3 input modes:**
- Manual sliders — drag V14 toward -7 to trigger fraud detection
- Paste raw values — copy any row from `creditcard.csv`
- Pre-built samples — load a legit / fraud / borderline example

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Run the notebooks (in order)
```bash
jupyter notebook Phase1_EDA_CreditCardFraud.ipynb
jupyter notebook Phase2_Preprocessing_SMOTE.ipynb
jupyter notebook Phase3_Model_Training.ipynb
jupyter notebook Phase4_Model_Evaluation.ipynb
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

> **Auto-train mode:** If `.pkl` model files are not present, the app will automatically train the XGBoost model from `creditcard.csv` on first launch (~30–60 seconds).

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py` → click **Deploy**
4. Get a public URL: `https://your-name-fraud-shield.streamlit.app`

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── 📓 Phase1_EDA_CreditCardFraud.ipynb       # EDA — 8 plots
├── 📓 Phase2_Preprocessing_SMOTE.ipynb       # Scaling + SMOTE — 3 plots
├── 📓 Phase3_Model_Training.ipynb            # Train 3 models — 5 plots
├── 📓 Phase4_Model_Evaluation.ipynb          # Full evaluation — 8 plots
│
├── 🌐 app.py                                 # Streamlit dashboard (775 lines)
├── 📋 requirements.txt                       # Python dependencies
│
├── 🤖 model_xgboost_best.pkl                 # Saved XGBoost model
├── ⚖️  scaler.pkl                            # Fitted StandardScaler
├── 🎯 best_threshold.pkl                     # Optimal threshold (0.85)
│
├── 📄 CreditCardFraud_ProjectReport.pdf      # 2-page project report
└── 📖 README.md                              # This file
```

---

## 🛠️ Tech Stack

```
Language    : Python 3.9
Models      : Logistic Regression, Random Forest, XGBoost
Balancing   : SMOTE (imbalanced-learn)
Evaluation  : AUC-ROC, Recall, F1, MCC, Confusion Matrix
Deployment  : Streamlit
Libraries   : Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Visualizations Generated

| Phase | Plot | Description |
|-------|------|-------------|
| EDA | Class distribution | Bar + pie — 578:1 imbalance |
| EDA | Amount analysis | Fraud vs legit comparison + boxplot |
| EDA | Time analysis | Fraud frequency by hour |
| EDA | Feature correlation | Pearson correlation of V1–V28 with Class |
| EDA | Correlation heatmap | Top feature inter-correlations |
| EDA | Feature distributions | Density plots — top 9 features |
| EDA | Violin plots | Spread of top 6 features by class |
| Preprocessing | Scaling effect | Amount & Time before/after StandardScaler |
| Preprocessing | SMOTE comparison | Class balance before vs after |
| Preprocessing | SMOTE PCA scatter | Real vs synthetic fraud sample placement |
| Training | Model comparison | All metrics — side-by-side bar chart |
| Training | Confusion matrices | TP/TN/FP/FN for all 3 models |
| Training | ROC curves | AUC comparison across models |
| Training | PR curves | Precision-Recall — better for imbalance |
| Training | Feature importance | Top 15 features — RF vs XGBoost |
| Evaluation | Metrics heatmap | 7 metrics × 3 models |
| Evaluation | Cost analysis | Financial impact of FN vs FP |
| Evaluation | Threshold tuning | F1/Recall/Precision vs threshold sweep |

---

## 💡 Why These Choices?

**Why XGBoost?**
Gradient boosting builds trees sequentially — each corrects the previous one's errors. It consistently outperforms Random Forest on tabular data and is the industry standard for fraud detection.

**Why SMOTE instead of random oversampling?**
Random oversampling just copies existing fraud rows — the model memorizes them. SMOTE creates new synthetic samples by interpolating between real fraud cases, giving the model more diverse patterns to learn from.

**Why tune the threshold?**
The default 0.5 threshold treats FP and FN equally. In fraud detection, a missed fraud (FN) costs ~$150 while a false alarm (FP) costs ~$5. Lowering the threshold to 0.85 captures this asymmetry and improves F1 from 0.40 to 0.73.

**Why MCC over F1?**
F1-Score ignores true negatives. MCC evaluates all four cells of the confusion matrix (TP, TN, FP, FN) equally — making it the most reliable metric when one class vastly outnumbers the other.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle
- Research paper: Andrea Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification"

---

<p align="center">Built with Python · Scikit-learn · XGBoost · Streamlit</p>
