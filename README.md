<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🧬%20NovaGen%20Health%20Classifier&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Binary%20Classification%20of%20Individual%20Health%20Risk%20for%20Biomedical%20Research&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

**NovaGen Health Classifier** is a supervised binary classification system built for **NovaGen Research Labs** — a leading biomedical research institute conducting large-scale population health studies. The model classifies individuals as **healthy (0)** or **unhealthy / at health risk (1)** based on physiological measurements, lifestyle factors, and medical history.

> **Clinical Motivation:** In biomedical research, missing a high-risk patient is far more dangerous than a false alarm. This project prioritizes **Recall** as the primary evaluation metric alongside Accuracy.

**Key Use Cases:**
- Selecting eligible participants for clinical trials and longitudinal studies
- Stratifying populations for risk-based analysis and outcome comparison

---

## 📂 Dataset

| Property | Value |
|:---|:---|
| Source | NovaGen Research Labs (Observational Studies) |
| Records | 9,800 unique individuals |
| Features | 22 (numerical + categorical/one-hot encoded) |
| Target | Binary — `0` = Healthy, `1` = Unhealthy / At Risk |
| Missing Values | None |
| Class Balance | ~52% Unhealthy, ~48% Healthy |

**Feature Categories:**
- **Physiological:** Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate
- **Lifestyle:** Sleep Hours, Exercise Hours, Water Intake, Stress Level, Smoking, Alcohol
- **Medical History:** Diet, MentalHealth, PhysicalActivity, MedicalHistory, Allergies
- **One-Hot Encoded:** Diet_Type (Vegan/Vegetarian), Blood_Group (AB/B/O)

---

## 🔄 Pipeline Workflow

```
Raw CSV → EDA & Null Check → Feature Selection → Train-Test Split → StandardScaler → Model Training → Evaluation (Recall-First) → Best Model Selection
```

1️⃣ **Data Loading & EDA** — Loaded 9,800 records, verified zero missing values, explored distributions across all 22 features.

2️⃣ **Preprocessing** — Applied `StandardScaler` for Logistic Regression and KNN. Tree-based models used raw unscaled features.

3️⃣ **Train-Test Split** — 80/20 stratified split → 7,890 training / 1,910 test samples.

4️⃣ **Model Training** — Trained 5 classifiers: Logistic Regression, KNN, Random Forest, Gradient Boosting, and a Soft Voting Ensemble.

5️⃣ **Evaluation** — Compared Accuracy + Recall across all models. Recall prioritized to minimize missed high-risk cases.

6️⃣ **Best Model Selection** — Random Forest (200 estimators) emerged as the top performer on both metrics.

---

## 🤖 Models

### 1️⃣ Logistic Regression (L2 Regularization)

```python
LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
```
- Baseline linear model with ridge regularization
- Trained on StandardScaler-transformed features
- Interpretable coefficients for clinical explainability

### 2️⃣ K-Nearest Neighbors (k=5, Euclidean)

```python
KNeighborsClassifier(n_neighbors=5, metric="euclidean")
```
- Instance-based learner, sensitive to feature scale
- Requires StandardScaler — performs well on dense health data
- Good balance of precision and recall

### 3️⃣ Random Forest ⭐ Best Model

```python
RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
```
- Ensemble of 200 decision trees via bagging
- Handles non-linear feature interactions natively
- Robust to outliers in physiological measurements
- **Highest Recall of 95.88%** — critical for missing zero high-risk patients

### 4️⃣ Gradient Boosting

```python
GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
```
- Sequential boosting corrects prior model errors
- Conservative max_depth=3 to prevent overfitting on health data
- Strong runner-up at 93.03% accuracy

### 5️⃣ Soft Voting Ensemble (LR + KNN + RF)

```python
VotingClassifier(estimators=[("lr", ...), ("knn", ...), ("rf", ...)], voting="soft")
```
- Combines probability estimates from 3 diverse base learners
- Reduces individual model variance
- Slightly lower than standalone RF due to LR's lower performance dragging the ensemble

---

## 📊 Results

| Model | Precision | Recall | F1 Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.82 | 0.83 | 0.82 | 81.41% |
| KNN (k=5) | 0.89 | 0.88 | 0.89 | 88.32% |
| 🏆 **Random Forest** | **0.93** | **0.96** | **0.94** | **93.82%** |
| Gradient Boosting | 0.92 | 0.95 | 0.93 | 93.04% |
| Soft Voting Ensemble | 0.91 | 0.93 | 0.92 | 91.57% |

> 🎯 **Primary Metric:** Recall — minimizing false negatives (missed high-risk patients) is the core clinical objective.

---

## 🔍 Key Insights

- 🏆 **Random Forest** achieved the highest Recall of **95.88%** — the most critical metric for biomedical risk screening
- 📈 **Ensemble methods** (RF, GB, Voting) consistently outperformed linear models by **10–12% in accuracy**
- ⚠️ **Logistic Regression**, despite being the simplest model, still achieved 81.4% accuracy — indicating strong linear separability in some features
- 🔬 **Glucose Level, BMI, and Blood Pressure** are expected dominant features, consistent with established medical risk indicators
- 📊 The dataset had **zero missing values** — unusually clean for medical data, enabling direct modeling without imputation
- 🎯 **Soft Voting Ensemble underperformed standalone RF** — LR's weaker probabilities dragged down the ensemble's soft vote

---

## 🗂️ Repository Structure

```
NovaGen-Health-Classifier/
│
├── NovaGen.ipynb              
├── novagen_dataset.csv        
└── README.md                            
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/NovaGen-Health-Classifier.git
cd NovaGen-Health-Classifier

# Install dependencies
pip install numpy pandas scikit-learn jupyter

# Launch notebook
jupyter notebook NovaGen.ipynb
```
---

## 🧠 Key Learnings

- **Recall > Accuracy** in medical classification — missing a sick patient has higher cost than a false alarm
- Tree-based ensembles (RF, GB) naturally handle mixed feature types (physiological + lifestyle + categorical) without scaling
- Soft voting ensembles are only as strong as their weakest member — careful base learner selection is critical
- Feature engineering (one-hot encoding for Blood Group and Diet Type) is essential for categorical health attributes
- StandardScaler is mandatory for distance-based models (KNN) and gradient-based models (Logistic Regression)

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| scikit-learn | All ML models + preprocessing |
| Pandas | Data loading, manipulation, EDA |
| NumPy | Numerical computations |
| Jupyter Notebook | Development & experimentation |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
