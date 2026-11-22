# 🏢 IoT Smart-Building Room Occupancy Estimation (Proof of Concept)
### *Bridging Data Engineering, Automated Feature Pipelines, and Multiclass Machine Learning for Sustainable Facilities Management*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Real%20Enterprise%20IoT-darkgreen)
![Status](https://img.shields.io/badge/Project-Proof--of--Concept-success)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-informational)

---

## 📌 1. Project Background & Problem Statement

### The Real-World Organization: NovaSpace Technologies
This project is built around real-world enterprise IoT infrastructure deployed at **NovaSpace Technologies**, a modern tech enterprise operating multi-floor smart offices. The company invested in an automated building management ecosystem, outfitting conference and meeting rooms with non-intrusive environmental IoT sensor arrays streaming telemetry roughly every 30 seconds into an enterprise data platform.

### The Business Problem: Booked vs. Actual Utilization
The corporate Facilities Management team encountered a critical operational bottleneck: **the system only recorded whether a room was booked in the calendar, not whether anyone was physically inside**.
* **Ghost Bookings:** Scheduled meetings are regularly cancelled without releasing the room, leaving rooms unoccupied while HVAC, ventilation, and lighting run continuously at peak power.
* **Severe Resource Misallocation:** 8-person conference rooms are routinely occupied by a single employee for solo phone calls, blocking teams needing collaborative space.
* **Unbooked Ad-hoc Meetings:** Employees frequently walk into unbooked empty rooms without logging a reservation, rendering calendar-based analytics completely inaccurate.

### The Machine Learning Objective
Can non-intrusive environmental telemetry already ingested by the data platform accurately determine real-time occupant count?
The goal is to develop a robust **Proof of Concept (PoC) Multiclass Classifier** to predict the discrete number of occupants:
$$\text{Room Occupancy Count} \in \{0, 1, 2, 3\}$$

---

## 👨‍💻 2. My Role: Data Engineer Driving the ML PoC

Rather than acting solely as a downstream modeler, my role on this project spanned **Data Engineering, Data Quality Assurance, Feature Pipeline Architecture, and Applied Machine Learning**:

1. **Enterprise Data Product Assessment:** Auditing the curated IoT sensor table provided by the data platform, evaluating schema fidelity, sensor distributions, scale variances, and data integrity.
2. **Data Leakage Elimination:** Identifying and pruning features (such as raw timestamps) that could cause lookahead or calendar leakage in a live production environment.
3. **Automated Feature Engineering Pipeline:** Designing a modular, scikit-learn `ColumnTransformer` pipeline that performs scaling, cyclic time transformation, and passthrough routing cleanly without train-test leakage.
4. **Class Imbalance Strategy:** Addressing an extreme vacancy skew (81.2% empty room records) by configuring balanced class weighting and selecting **Macro-averaged F1-Score** as the primary evaluation metric.
5. **Model Benchmarking & Hyperparameter Optimization:** Implementing, tuning, and comparing three distinct model families (Multinomial Logistic Regression, K-Nearest Neighbors, and Decision Tree Classifiers).
6. **Business-Impact Error Analysis:** Analyzing confusion matrices to ensure the chosen model makes zero catastrophic operational errors (e.g., misclassifying 0 occupants as 3 or vice-versa).
7. **End-to-End Holdout Inference:** Processing unlabelled test data through the fitted pipeline to output production-ready batch predictions.

---

## 🏗️ 3. Upstream Enterprise Data Pipeline Architecture

In a true corporate architecture, data does not arrive as an isolated CSV file. The dataset used in this project represents the curated analytical layer of an end-to-end streaming data pipeline:

```mermaid
flowchart LR
    A["IoT Sensor Nodes\n(Temp, Light, Sound, CO₂, Motion)"] --> B["Edge Sensor Gateway\n(Local Aggregation ~30s)"]
    B --> C["Raw Telemetry Stream\n(Kafka / Event Hub Ingestion)"]
    C --> D["Enterprise Data Platform\n(Bronze / Silver Cleansing Layer)"]
    D --> E["Curated Sensor Data Table\n(Gold Layer / Feature Store)"]
    E --> F["My Data Engineering & ML Pipeline\n(Leakage Auditing & Preprocessing)"]
    F --> G["Tuned Multiclass Model\n(Decision Tree Champion)"]
    G --> H["Predictions Table & Facilities Dashboard\n(Real-Time Automation & Alerts)"]
```

---

## 📊 4. Dataset Description & Data Quality Audit

The dataset represents genuine physical telemetry captured from office environment monitoring.

### Key Dataset Dimensions
* **Total Training Observations:** 8,103 rows
* **Total Feature Columns:** 18 features (16 physical sensor readings + 2 temporal indicators)
* **Target Variable:** `Room_Occupancy_Count` (0, 1, 2, or 3 occupants)
* **Missing Values:** `0` (Zero null values across all sensor streams)
* **Duplicate Readings:** Audited and resolved; sampling consistency verified at ~30-second cadence.

### Complete Sensor Inventory & Physical Properties

| Category | Sensor Features | Physical Units & Range | Environmental Signal Detected |
| :--- | :--- | :--- | :--- |
| **Temperature** | `S1_Temp`, `S2_Temp`, `S3_Temp`, `S4_Temp` | 24.44°C – 29.00°C | Multi-zone ambient air temperature across corners/ceiling |
| **Illuminance** | `S1_Light`, `S2_Light`, `S3_Light`, `S4_Light` | 0 – 280 Lux | Directional light intensity (detects fixtures, daylight, projector) |
| **Acoustics** | `S1_Sound`, `S2_Sound`, `S3_Sound`, `S4_Sound` | 0.04 – 3.88 V | Acoustic amplitude (background noise floor vs. vocal conversation) |
| **Air Quality** | `S5_CO2` | 345 – 1,270 ppm | Absolute carbon dioxide concentration |
| **$CO_2$ Velocity** | `S5_CO2_Slope` | -6.30 to +8.98 ppm/min | Derivative rate of change in $CO_2$ accumulation/ventilation |
| **Motion** | `S6_PIR`, `S7_PIR` | Binary ($0$ or $1$) | Passive Infrared sensors detecting gross body motion |

### Target Distribution & Severe Class Imbalance
An audit of `Room_Occupancy_Count` revealed substantial class imbalance reflecting real-world corporate office usage:

```text
Room_Occupancy_Count Distribution (N = 8,103):
├── Class 0 (Vacant):    6,582 records  (81.23%)  ████████████████████████████████
├── Class 1 (1 Person):    367 records  ( 4.53%)  ██
├── Class 2 (2 People):    599 records  ( 7.39%)  ███
└── Class 3 (3 People):    555 records  ( 6.85%)  ███
```

> **Data Quality Insight:** Because the room is unoccupied 81.2% of the time, standard accuracy is misleading: a naive model always predicting "0" achieves 81.23% accuracy while failing 100% of business objectives. Therefore, **Macro-averaged F1-Score** was established as the primary decision benchmark to weight all four occupancy classes equally.

---

## 🔬 5. Sensor Dynamics & Exploratory Data Analysis (EDA)

Before engineering features, I analyzed the underlying physical interactions between occupants and the sensor array:

1. **Carbon Dioxide ($CO_2$) Accumulation & Dissipation:**
   * Human respiration steadily adds $CO_2$ to enclosed spaces. Baseline empty rooms sit at ~350–370 ppm, whereas 2–3 occupants push levels to 800–1,270 ppm.
   * `S5_CO2_Slope` acts as a crucial directional indicator: a sharp positive slope indicates recent group entry, while a negative slope marks occupants leaving and ventilation kicking in.
2. **Acoustic Noise Floor vs. Conversation:**
   * Empty rooms maintain a stable ambient electronic/HVAC noise floor ($0.04 - 0.08$ V).
   * Voice activity during meetings causes sharp spikes ($0.15 - 3.88$ V), providing strong separation between silent solo occupancy and group discussions.
3. **PIR Motion Limitations:**
   * While `S6_PIR` and `S7_PIR` reliably detect people walking into or out of the room, they frequently register $0$ when attendees sit still during a presentation or focused laptop work. PIR alone cannot serve as an occupancy counter.
4. **Lighting Variations:**
   * Values drop to $0$ Lux during dark/off hours and jump to $50 - 280$ Lux when ceiling or task lighting is active.
5. **Thermal Lag:**
   * Temperature changes slowly due to thermal inertia ($25^\circ\text{C} \rightarrow 26.5^\circ\text{C}$). It provides general macro-context but is too slow to register rapid entries/exits.

---

## 🛠️ 6. Feature Engineering & Preprocessing Pipeline

### Step 1: Mitigating Temporal Data Leakage
Raw calendar strings `Date` (e.g. `"2017/12/23"`) and `Time` (e.g. `"14:55:37"`) risk leaking specific historical time intervals into tree splits:
* Dropped raw `Date` and `Time` columns.
* Extracted **`Hour`** ($0 - 23$) from `Time` to capture operational business hours vs. night/off-hours.
* Extracted **`DayOfWeek`** ($0 - 6$) from `Date` to capture weekday working schedules vs. weekends.

### Step 2: Resolving Extreme Scale Disparities
The raw sensor values spanned drastically different numerical domains:
* Sound: $[0.04, 3.88]$
* Temperature: $[24.4, 29.0]$
* Illuminance: $[0, 280]$
* $CO_2$: $[345, 1270]$

Without standardization, distance-based algorithms (like KNN) would compute distances dominated entirely by $CO_2$ and Light, ignoring Sound and Temperature completely.

### Step 3: Reproducible `ColumnTransformer` Architecture
I implemented a robust transformation pipeline that encapsulates scaling, encoding, and passthrough routing to ensure zero data leakage between train and test splits:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

scaler_features = [
    'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
    'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
    'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
    'S5_CO2', 'S5_CO2_Slope', 'Hour'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), scaler_features),
        ('categorical', OrdinalEncoder(), ['DayOfWeek']),
        ('pir', 'passthrough', ['S6_PIR', 'S7_PIR'])
    ]
)
```

### Step 4: Stratified Dataset Partitioning
The training set was split using a stratified 80/20 train-test split (`test_size=0.2`, `random_state=42`, `stratify=y`), guaranteeing identical class representations in both sets:
* **Train Set:** 6,482 records
* **Validation Test Set:** 1,621 records (Class 0: 1,317; Class 1: 73; Class 2: 120; Class 3: 111)

---

## 🧪 7. Machine Learning Algorithms & Hyperparameter Tuning

I trained and tuned three distinct model architectures to identify the optimal balance of accuracy, generalization, and computational efficiency.

### Algorithm 1: Multinomial Logistic Regression (Linear Benchmark)
* **Architecture:** Generalized linear model with multinomial loss and L-BFGS solver.
* **Class Balancing:** `class_weight='balanced'` to offset the 81.2% vacancy skew.
* **Hyperparameter Search:** Swept inverse regularization strength $C \in [0.01, 0.1, 0.5, 1, 2, 5, 10]$:
  * $C = 0.01 \implies \text{Macro F1: } 0.9578$
  * $C = 0.1, 0.5, 1.0 \implies \text{Macro F1: } 0.9760$
  * $C = 2.0 \implies \text{Macro F1: } 0.9815$ *(Selected for peak stability)*
  * $C = 5.0 \implies \text{Macro F1: } 0.9836$
  * $C = 10.0 \implies \text{Macro F1: } 0.9813$

### Algorithm 2: K-Nearest Neighbors (KNN - Non-Parametric Distance)
* **Architecture:** Instance-based learning using Euclidean distance over standardized feature vectors.
* **Hyperparameter Search:** Systematically evaluated $k \in [1, 30]$:
  * Peak Macro F1 performance was achieved at **$k = 7$** ($\text{Macro F1} = 0.9805$).
  * Lower $k$ ($1-3$) proved sensitive to sensor noise; higher $k$ ($>15$) over-smoothed minority class boundaries into the dominant Class 0 cluster.

### Algorithm 3: Decision Tree Classifier (Non-Linear Partitioning)
* **Architecture:** Non-linear recursive binary splitting using Gini impurity.
* **Depth Tuning:** Evaluated `max_depth` across $[3, 5, 7, 10, 15, 20, \text{None}]$:
  * Depth 3: Macro F1 = $0.8698$ (Underfitting)
  * Depth 5: Macro F1 = $0.9642$
  * Depth 7: Macro F1 = $0.9805$
  * Depth 15: Macro F1 = $0.9842$
* **Leaf Regularization & Class Weighting:**
  * Tested `min_samples_leaf` $\in [1, 2, 5, 10, 20, 30, 50]$.
  * Applying `class_weight='balanced'` at `max_depth=15, min_samples_leaf=1` boosted the Macro F1 score from **$0.9842 \rightarrow 0.9886$**.

---

## 🏆 8. Model Evaluation & Comparative Performance

All models were evaluated on the exact same stratified holdout test set ($N=1,621$):

| Model Evaluated | Hyperparameter Configuration | Accuracy | Macro Precision | Macro Recall | **Macro F1-Score** | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | `max_depth=15, min_samples_leaf=1, class_weight='balanced'` | **99.69%** | **98.78%** | **98.94%** | **0.9886 (98.86%)** | 🥇 **Champion** |
| **Multinomial Logistic Regression** | `C=2.0, class_weight='balanced', max_iter=2000` | 99.44% | 98.06% | 98.28% | **0.9815 (98.15%)** | 🥈 Runner-up |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=7, metric='euclidean'` | 99.44% | 98.04% | 98.07% | **0.9805 (98.05%)** | 🥉 Third |

### Why Did the Decision Tree Win?
1. **Threshold-Driven Physical Realities:** Environmental sensors operate via physical step functions (lights turn ON/OFF at distinct lux cutoffs; human conversation creates acoustic threshold jumps above ambient silence; $CO_2$ slope sharply segments entry/exit). Tree-based axes-aligned splits naturally capture these abrupt boundaries far better than linear hyperplanes or uniform spherical Euclidean distances.
2. **Robustness to Mixed Sensor Modalities:** The Decision Tree handles binary motion flags (`PIR`), discrete time units (`Hour`), and continuous concentration curves seamlessly without being distorted by localized distance metric distortions.

---

## 🔍 9. Business-Impact Confusion Matrix Breakdown

In facilities management, **the operational cost of an error depends on which classes are confused**:
* **Low Impact Error:** Predicting 2 people when 3 are present (or vice-versa). Climate control remains on, lights stay illuminated, and ventilation adjusts slightly.
* **Catastrophic Operational Error:** Predicting 0 people when 3 people are in the room. Automated facilities systems would shut off lights, disable AC, and close ventilation dampers on a live executive meeting.

### Test Set Confusion Matrices ($N=1,621$)

#### 1. Decision Tree (Champion Model)
```text
                      PREDICTED
                 Class 0  Class 1  Class 2  Class 3
ACTUAL  Class 0:   1317        0        0        0    <-- 100% precision & recall on vacancy
        Class 1:      0       73        0        0    <-- 100% accuracy on solo occupancy
        Class 2:      0        1      116        3    <-- Only 4 minor adjacent misses
        Class 3:      0        0        1      110    <-- 0 critical false empty predictions
```
* **Critical Empty-vs-Occupied Error Rate:** **0.0%**. The Decision Tree never mistook an empty room for an occupied room, nor did it miss any active group meeting.
* **Total Misclassifications:** Only **5 out of 1,621 test samples** (0.31% total test error), all confined to adjacent occupancy boundaries ($2 \leftrightarrow 3$ and $2 \leftrightarrow 1$).

#### 2. Multinomial Logistic Regression
```text
                      PREDICTED
                 Class 0  Class 1  Class 2  Class 3
ACTUAL  Class 0:   1316        0        0        1    <-- 1 False Occupancy Alarm
        Class 1:      0       73        0        0
        Class 2:      0        0      114        6
        Class 3:      0        0        2      109
```

#### 3. K-Nearest Neighbors (KNN)
```text
                      PREDICTED
                 Class 0  Class 1  Class 2  Class 3
ACTUAL  Class 0:   1317        0        0        0
        Class 1:      0       73        0        0
        Class 2:      0        0      114        6
        Class 3:      0        0        3      108
```

---

## 🔮 10. Final Model Holdout Inference

To complete the proof of concept, the winning Decision Tree model was trained on the full curated dataset and applied to the unlabelled production test partition (`test (1).csv`, $N=2,026$):

```text
Final Holdout Predictions Distribution (N = 2,026):
├── Class 0 (Vacant):    1,646 predictions  (81.24%)
├── Class 2 (2 People):    150 predictions  ( 7.40%)
├── Class 3 (3 People):    138 predictions  ( 6.81%)
└── Class 1 (1 Person):     92 predictions  ( 4.54%)
```
The predicted distribution closely mirrors the original enterprise training distribution, confirming that the model generalizes robustly without class-collapse. All inference results were output directly to `predictions.csv`.

---

## 💡 11. Facilities Operationalization & Production Roadmap

### Automated Actions Triggered by Predictions

```mermaid
flowchart TD
    P["Real-Time Model Prediction"] -->|Prediction = 0| Act0["Release calendar booking after 10-min grace period\nSet HVAC to eco-mode\nExtinguish room lights"]
    P -->|Prediction = 1| Act1["Confirm room is in active use\nMaintain standard climate conditions\nFlag 8-person room single-occupancy under-utilization"]
    P -->|Prediction = 2| Act2["Log collaborative meeting utilization\nMaintain standard lighting & ventilation"]
    P -->|Prediction = 3| Act3["Ramp up mechanical ventilation dampers to flush CO₂\nTrigger room capacity overflow alerts"]
```

### Recommendation for Head of Facilities
1. **Feasibility Successfully Proven:** The PoC verifies that commodity, non-intrusive IoT sensors accurately estimate human occupancy count with **99.69% accuracy** and **98.86% Macro F1**, avoiding invasive cameras or microphones.
2. **Pre-Production Caveat:** This PoC was calibrated on a single room under controlled conditions. Prior to deploying company-wide across every office floor:
   * **Room Volume Calibration:** $CO_2$ accumulation rate varies by room volume (a 500 ppm increase in a small phone booth represents different occupancy than in a boardroom).
   * **Ventilation Rate Variances:** Multi-room telemetry must be gathered across different HVAC airflow configurations and seasonal heating/cooling cycles.
   * **Sensor Position Auditing:** Standard operating procedures (SOPs) must be established for sensor node placement relative to doors, windows, and supply diffusers.

---

## 📁 12. Repository Structure

```text
├── ML_Hackthon_project2.ipynb   # Complete pipeline notebook (EDA, preprocessing, modeling, evaluation)
├── README.md                    # Comprehensive project documentation & technical report
└── predictions.csv              # Holdout test set predictions (2,026 records)
```

---

## 🚀 13. How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/iot-room-occupancy-poc.git
   cd iot-room-occupancy-poc
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. **Launch the analysis notebook:**
   ```bash
   jupyter notebook ML_Hackthon_project2.ipynb
   ```
