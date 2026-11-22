# NorthStar Bank: Term Deposit Subscription Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen.svg)]()

> An end-to-end Machine Learning classification system designed to transition retail banking marketing from broad, blind telemarketing to high-propensity, targeted customer acquisition.

---

## 📌 Executive Summary & HR Story

> *"At NorthStar Bank, Marketing was cold-calling every customer in the database for every Term Deposit campaign — an expensive approach that burned call-center hours and fatigued uninterested clients. I built a supervised classification pipeline to score customers as Likely vs. Unlikely to subscribe, using demographic, financial, and historical campaign data.*
>
> *A critical early finding was the presence of `duration` (call duration in seconds) in the raw dataset. While statistically predictive, call duration cannot be known before dialing; keeping it would introduce severe data leakage and create a model that looks remarkable in a notebook but fails in production. I removed it immediately.*
>
> *Due to an 88:12 class imbalance (only ~11.7% conversion rate), baseline accuracy was misleading. I evaluated Logistic Regression, K-Nearest Neighbors, and Decision Trees using Macro F1, Recall, Precision, and Confusion Matrices. I selected the **Cost-Sensitive Decision Tree (Depth=7, Class-Balanced)**, which achieved the highest **Macro F1 (0.6606)** and captured over 52% of conversions while cutting uninterested calls by ~87%. In production, this allows Marketing to prioritize the top-propensity call list rather than dialing blind."*

---

## 🏢 Business Context & Problem Statement

### Company Profile
**NorthStar Bank** is a diversified financial institution offering retail savings accounts, consumer credit, mortgages, and investment products, including fixed-term deposits.

### The Business Challenge
Whenever Marketing launched a Term Deposit campaign, the call center dialed every single contact in the database:
- **High Operational Costs:** Call-center agent capacity was saturated with unproductive leads.
- **Customer Friction:** Uninterested customers received repetitive telemarketing calls.
- **Opportunity Loss:** High-value customers ready to convert received delayed outreach.

### Machine Learning Formulation
- **Task Type:** Supervised Binary Classification.
- **Target Variable ($y$):**
  - `1` (`yes`): Customer subscribes to a Term Deposit.
  - `0` (`no`): Customer does not subscribe.
- **Business Operationalization:** Transition from exhaustive calling to a **Ranked / Prioritized Calling Queue**, where call agents dial high-probability leads first, maximizing conversion per agent hour.

---

## 📊 Dataset Overview & Audit

The dataset reflects retail banking customer demographics, previous campaign outcomes, and socio-economic touchpoints.

| Metric / Dimension | Training Set | Test Set (Inference) |
| :--- | :--- | :--- |
| **Total Rows** | 36,168 | 9,043 |
| **Total Raw Columns** | 17 | 17 (includes unique customer `id`) |
| **Missing / Null Values** | 0 (`.isna().sum() == 0`) | 0 |
| **Duplicate Records** | 0 | 0 |
| **Class Distribution ($y$)** | `no`: 31,937 (88.3%)<br>`yes`: 4,231 (11.7%) | Unlabeled (Target for prediction) |

### Key Data Observations
- **Severe Class Imbalance:** ~88.3% non-subscribers vs. ~11.7% subscribers. A naive majority-class classifier yields 88.3% accuracy while offering zero operational value.
- **Implicit Missingness:** While standard null checks returned zero, attributes such as `job` (224 entries), `education` (1,488 entries), `poutcome`, and `contact` contain explicit `"unknown"` strings, preserved to prevent data loss.

---

## 🔍 Exploratory Data Analysis & Critical Findings

### 1. The Call Duration Trap (Data Leakage)
- The raw dataset includes a `duration` attribute (call duration in seconds).
- **Domain Insight:** Call duration is strictly recorded *post-call*. At the moment Marketing generates daily call sheets, call duration is unknown ($t=0$).
- **Engineering Decision:** `duration` was permanently dropped from feature matrix $X$ prior to modeling to prevent catastrophic target leakage.

### 2. Campaign Fatigue & Recency Dynamics
- `previous` (number of contacts prior to this campaign) and `poutcome` (outcome of past campaigns) exhibited strong directional signals: customers with previous campaign `success` convert at a markedly higher rate.
- Repeated dials within the current campaign (`campaign`) hit diminishing returns quickly; excessive contacts correlate with lower conversion.

---

## ⚙️ Feature Engineering & Preprocessing Pipeline

### Feature Engineering
Three domain-specific features were engineered to capture historical interaction strength:

1. `was_contacted_before`: Binary flag (`previous > 0`) differentiating new prospects from existing campaign relationships.
2. `previous_success`: Binary indicator (`poutcome == 'success'`) isolating customers with verified historical conversion affinity.
3. `total_contacts`: Additive cumulative contact count (`campaign + previous`) capturing cumulative campaign pressure.

### Preprocessing Architecture (`ColumnTransformer`)
Features were partitioned and transformed without data leakage:

```text
Raw Features (18 Predictors)
 │
 ├── Numerical Features (9) ─────────► StandardScaler()
 │   ['age', 'balance', 'day', 'campaign', 'pdays',
 │    'previous', 'was_contacted_before',
 │    'previous_success', 'total_contacts']
 │
 └── Categorical Features (9) ───────► OneHotEncoder(handle_unknown='ignore')
     ['job', 'marital', 'education', 'default',
      'housing', 'loan', 'contact', 'month', 'poutcome']
 │
 └── Output Matrix: 53 Preprocessed Feature Columns
```

### Validation Strategy
- **Train-Test Split:** 80% Train (28,934 samples) / 20% Test (7,234 samples).
- **Stratification:** Stratified by target $y$ (`stratify=y`, `random_state=42`) to guarantee identical class distributions across both splits.

---

## 🧪 Model Development & Hyperparameter Tuning

Three distinct model families were implemented, tuned, and evaluated on the holdout test set:

### 1. Regularized Logistic Regression
- **Purpose:** Linear decision baseline with calibrated probabilities.
- **Tuning:** Sweep across inverse regularization parameter $C \in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]$ with `class_weight='balanced'` and `max_iter=2000`.
- **Optimal Parameter:** $C = 5.0$ yielded peak Macro F1 of `0.6042`.

### 2. K-Nearest Neighbors (KNN)
- **Purpose:** Non-parametric instance-based classifier.
- **Tuning:** Iterated $k \in [1, 30]$ on standard-scaled features.
- **Selected Parameter:** $k = 5$ (Macro F1 = `0.6260`).

### 3. Decision Tree Classifier
- **Purpose:** Non-linear rule-based model capable of capturing complex interactions without distance distortion.
- **Tuning Progression:**
  - `max_depth` sweep: tested values $[3, 5, 7, 10, 15, 20, \text{None}]$; depth 7–10 performed best.
  - `class_weight`: testing `'balanced'` increased Macro F1 from `0.6264` to `0.6425`.
  - Joint grid on `max_depth` $\times$ `min_samples_leaf`: depth 7 with leaf constraints yielded top stability.
  - Criteria comparison: Evaluated `gini` vs. `entropy` vs. `log_loss`.
- **Selected Configuration:** `DecisionTreeClassifier(max_depth=7, class_weight='balanced', random_state=42)`.

---

## 📈 Model Evaluation & Comparison

Performance on the holdout test set ($N = 7,234$, with 6,388 actual 'no' and 846 actual 'yes'):

| Model Architecture | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro F1 Score | Confusion Matrix [TN, FP / FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** ($C=5.0$, balanced) | 74.94% | 25.91% | **61.47%** | 0.6042 | `[[4901, 1487], [326, 520]]` |
| **K-Nearest Neighbors** ($k=5$) | **88.64%** | **53.41%** | 22.22% | 0.6260 | `[[6224, 164], [658, 188]]` |
| **Decision Tree** ($\text{depth}=7$, balanced) 🏆 | 83.15% | 35.14% | 52.13% | **0.6606** | `[[5574, 814], [405, 441]]` |

### Business Trade-off Analysis: False Positives vs. False Negatives
In retail banking campaign economics, errors have asymmetric costs:

1. **False Positive (Type I Error):**
   - *Scenario:* Model predicts "Yes", customer says "No".
   - *Cost:* ~3–5 minutes of call-center time and phone charges.
2. **False Negative (Type II Error):**
   - *Scenario:* Model predicts "No", but customer would have deposited funds.
   - *Cost:* Loss of significant interest margin, long-term balance retention, and customer lifetime value.

### Why KNN Fails the Business:
Despite the highest accuracy (**88.64%**), KNN suffered an unacceptable **77.78% False Negative rate** (missed 658 out of 846 actual subscribers). It optimized accuracy by defaulting toward the majority class.

### Why Decision Tree Wins:
- Delivers the highest overall **Macro F1 score (0.6606)**.
- Captures **441 true subscribers** (Recall: 52.13%) while reducing non-converting calls from 6,388 down to just 814 (an **87.25% reduction in wasted calls**).
- Produces clear, explainable decision paths that compliance and marketing stakeholders can audit.

---

## 🚀 Final Production Model & Inference

1. **Retraining:** The final `DecisionTreeClassifier(max_depth=7, class_weight='balanced', random_state=42)` was fitted on all 36,168 records using the full pipeline with `handle_unknown='ignore'` enabled.
2. **Inference on Test Set:** Evaluated across the 9,043 unlabelled prospect records (`test.csv`).
3. **Generated Predictions (`predictions.csv`):**
   - Predicted Non-Subscribers (`no`): **7,500 (82.9%)**
   - Predicted Target Calls (`yes`): **1,543 (17.1%)**
4. **Output Verification:** Successfully produced 9,043 rows mapped to the schema `['id', 'prediction']`.

---

## 💡 Stakeholder Recommendations for Head of Marketing

```text
                               Daily Workflow
  ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
  │  Customer Database     │ ───► │ ML Scoring Pipeline     │ ───► │ Top Tier (17% of Base)  │
  │  (Unfiltered Prospects)│      │ (Decision Tree Model)   │      │ Dialed by Call Center   │
  └────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                                │
                                                                   Captures >50% of Total
                                                                   Conversions at 83% Lower
                                                                   Dialing Volume
```

1. **Implement Prioritized Outreach:**
   Stop calling 100% of customer lists. Focus phone campaigns strictly on the ~17% flagged by the model, capturing over half of total conversions while slashing telemarketing costs by over 80%.
2. **Omnichannel Re-allocation:**
   For customers flagged as `no`, avoid cold calls. Instead, transition them to low-cost digital touchpoints (email digests, mobile app notifications, in-app banners).
3. **Respect Operational Boundaries:**
   Never use call duration as an input criterion for prospect scoring; ensure real-time model scoring runs strictly on pre-call attributes.

---

## 🛠️ Repository Structure

```text
.
├── ML_Hackthon_project1.ipynb    # Complete EDA, preprocessing, training & tuning notebook
├── predictions.csv               # Final test set predictions (9,043 records)
├── README.md                     # Comprehensive project documentation
```

---

## 💻 How to Run

### Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Execution
Open and execute all cells in `ML_Hackthon_project1.ipynb` or run via Jupyter:
```bash
jupyter notebook ML_Hackthon_project1.ipynb
```
The notebook automatically outputs the preprocessed matrices, model benchmark tables, decision tree visualizations, and exports `predictions.csv`.
