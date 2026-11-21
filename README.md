# 🏢 IoT Smart-Building Room Occupancy Estimation (Proof of Concept)
### *Bridging Data Engineering and Applied Machine Learning for Sustainable Facilities Management*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Project-Proof--of--Concept-success)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-informational)

---

## 📌 Executive Summary & Business Context

At **NovaSpace Technologies**, meeting rooms across multiple offices are equipped with non-intrusive IoT sensor nodes streaming environmental telemetry every ~30 seconds into the centralized Enterprise Analytics Platform.

### The Business Challenge
The Facilities Management team currently faces an operational blind spot: **they only know if a room was booked, not if it is actually occupied**.
* **Ghost Bookings:** Scheduled meetings are frequently cancelled or no-shows occur, yet HVAC and lighting systems run continuously in empty spaces, driving unnecessary energy costs.
* **Under-utilization:** 8-person conference rooms are routinely occupied by single individuals.
* **Ad-hoc Usage:** Employees often hold spontaneous meetings in unreserved spaces.

### The Core Objective
Approaching this problem from the perspective of a **Data Engineer**, the goal is to evaluate an existing curated sensor data product and build a rigorous **Proof of Concept (PoC) Multiclass Classifier** to estimate exact room occupancy in real time:
$$\text{Room Occupancy Count} \in \{0, 1, 2, 3\}$$

---

## 🏗️ End-to-End Enterprise Data Architecture

Rather than treating the dataset as an isolated static CSV, the project models the data as an **enterprise data product** originating from an upstream streaming architecture:

```mermaid
flowchart LR
    A["IoT Sensor Nodes\n(Temp, Light, Sound, CO₂, PIR)"] --> B["Sensor Gateway\n(Edge Aggregator)"]
    B --> C["Raw Ingestion Layer\n(Kafka / Event Hub)"]
    C --> D["Enterprise Data Platform\n(Validation & Cleansing)"]
    D --> E["Curated Room Telemetry Table\n(Analytical Store)"]
    E --> F["ML Ingestion & Preprocessing\n(Feature Transformation)"]
    F --> G["PoC Classification Model\n(Decision Tree / KNN / Logistic)"]
    G --> H["Prediction Store & Real-Time Facilities Dashboard"]
```

---

## 📊 Dataset Overview & Data Quality Assessment

The curated dataset represents physical telemetry sampled at approximately 30-second intervals from a controlled conference room environment.

### 1. Schema & Characteristics
* **Total Records:** 8,103 observations
* **Features:** 18 input features (16 continuous/binary environmental sensors + 2 temporal features)
* **Target Feature:** `Room_Occupancy_Count` (Multiclass: 0, 1, 2, 3 occupants)
* **Missing Values:** `0` (Zero null values across all sensor streams)
* **Duplicate Rows:** Checked and verified for integrity

### 2. Sensor Inventory
| Category | Sensor Features | Physical Units & Typical Range | Sensor Description |
| :--- | :--- | :--- | :--- |
| **Temperature** | `S1_Temp`, `S2_Temp`, `S3_Temp`, `S4_Temp` | 24.44°C – 29.00°C | Multi-zone ambient air temperature sensors |
| **Illuminance** | `S1_Light`, `S2_Light`, `S3_Light`, `S4_Light` | 0 – 280 Lux | Directional light intensity (detects fixtures & sunlight) |
| **Acoustics** | `S1_Sound`, `S2_Sound`, `S3_Sound`, `S4_Sound` | 0.04 – 3.88 Volts / Amp | Ambient noise floor & vocal acoustic amplitude |
| **Air Quality** | `S5_CO2`, `S5_CO2_Slope` | 345 – 1270 ppm; $\Delta\text{ppm}/\Delta t$ | Absolute carbon dioxide concentration and rate of change |
| **Motion** | `S6_PIR`, `S7_PIR` | Binary flag ($0$ or $1$) | Passive Infrared (PIR) body motion sensors |

### 3. Target Class Distribution (Imbalance Profile)
The room remains vacant for the majority of monitoring hours, leading to a natural class imbalance:

| Occupancy Class | Sample Count | Proportion | Operational Meaning |
| :---: | :---: | :---: | :--- |
| **Class 0** | **6,582** | **81.23%** | Room is vacant (energy saving opportunity) |
| **Class 1** | **367** | **4.53%** | Solo occupant (phone call / focused work) |
| **Class 2** | **599** | **7.39%** | Small collaboration / 1-on-1 meeting |
| **Class 3** | **555** | **6.85%** | Full team meeting / group discussion |
| **Total** | **8,103** | **100.0%** | |

> **⚠️ Data Engineering Insight:** With **81.2%** of the records belonging to Class 0, naive classification accuracy is an uninformative metric (a trivial dummy baseline predicting "0" yields >81% accuracy). Consequently, **Macro-averaged F1-Score** is selected as the primary optimization metric, giving equal weight to each occupancy class.

---

## 🔍 Sensor Dynamics & Exploratory Data Analysis (EDA)

Understanding the physical domain dynamics is crucial before engineering model features:

1. **Carbon Dioxide Dynamics (`S5_CO2` & `S5_CO2_Slope`):**
   * Humans naturally exhale $CO_2$. While baseline ambient levels hover around 350–400 ppm, occupied rooms steadily climb toward 800–1270 ppm.
   * `S5_CO2_Slope` captures the velocity of change: a steep positive slope indicates people recently entered; a negative slope indicates room ventilation after departure.
2. **Acoustic Signature (`S1_Sound` – `S4_Sound`):**
   * Empty rooms exhibit a uniform background hum ($0.04 - 0.08$ V).
   * Spikes ($>0.50$ V) strongly signal conversation and human presence, distinguishing quiet 1-person work from multi-person collaborative discussion.
3. **Passive Infrared (`S6_PIR`, `S7_PIR`):**
   * PIR sensors detect thermal motion differentials. While excellent at registering entry, exit, or gesturing, they often drop to $0$ when occupants are seated quietly typing or listening. Thus, PIR alone is insufficient for occupancy count.
4. **Lighting Levels (`S1_Light` – `S4_Light`):**
   * Values drop to $0$ during dark/empty periods and surge during meeting hours. Different sensor values identify whether ceiling lights or specific desk fixtures are on.
5. **Temperature Response (`S1_Temp` – `S4_Temp`):**
   * Temperature exhibits thermal inertia—climbing slowly as body heat dissipates and HVAC systems compensate. It provides ambient context rather than instantaneous counts.

---

## ⚙️ Data Preprocessing & Leakage Prevention

### 1. Temporal Leakage Prevention
Raw `Date` and `Time` strings were intentionally **not** fed directly into the model to prevent temporal overfitting (e.g., memorizing specific calendar days or timestamps that do not generalize to future operational dates). Instead, they were engineered into domain-relevant cyclical/periodic indicators:
* `Hour = Time.dt.hour` (captures business operating hours vs. off-hours)
* `DayOfWeek = Date.dt.dayofweek` (captures weekday vs. weekend patterns)
* Raw `Date` and `Time` columns were removed from the training matrix.

### 2. Feature Scaling Strategy
Features exist on radically different scales:
* Temperature: $24 - 29$
* Acoustics: $0.04 - 3.88$
* Illuminance: $0 - 280$
* $CO_2$: $345 - 1270$

Distance-based algorithms (such as K-Nearest Neighbors) and gradient-based algorithms (such as Logistic Regression) would be dominated by raw $CO_2$ and Light values without standard scaling.

### 3. Modular Preprocessing Pipeline
A scikit-learn `ColumnTransformer` was designed to guarantee reproducible, leak-free transformations across train and test sets:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

preprocessor = ColumnTransformer([
    ('numerical', StandardScaler(), [
        'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
        'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
        'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
        'S5_CO2', 'S5_CO2_Slope', 'Hour'
    ]),
    ('categorical', OrdinalEncoder(), ['DayOfWeek']),
    ('pir', 'passthrough', ['S6_PIR', 'S7_PIR'])
])
```

* **Data Partitioning:** Stratified 80/20 train-test split (`test_size=0.2`, `random_state=42`, `stratify=y`) preserving class ratios ($N_{\text{train}} = 6,482$, $N_{\text{test}} = 1,621$).

---

## 🤖 Model Experimentation & Comparative Evaluation

Three diverse algorithmic families were tuned and benchmarked on identical stratified test partitions ($N=1,621$).

### 1. Multinomial Logistic Regression (Linear Baseline)
* **Optimization:** L-BFGS solver with balanced class weighting (`class_weight='balanced'`) to penalize minority misclassifications.
* **Regularization Tuning:** Evaluated $C \in [0.01, 0.1, 0.5, 1, 2, 5, 10]$. Selected $C = 2$ achieving peak validation stability.

### 2. K-Nearest Neighbors (Instance-Based Classifier)
* **Distance Metric:** Euclidean distance computed over standardized feature representations.
* **Neighbor Search ($k$):** Empirically swept $k \in [1, 30]$. Peak Macro F1 performance was achieved at $k = 7$ (balancing variance reduction against local boundary oversmoothing).

### 3. Decision Tree Classifier (Non-Linear Rule Induction)
* **Depth & Complexity Sweep:** Tested shallow ($depth = 3, 5$), intermediate ($depth = 7, 10, 15$), and unconstrained trees.
* **Regularization:** Compared `min_samples_leaf` ($1$ to $50$) and class weighting (`balanced` vs. `None`).
* **Selected Parameters:** `max_depth=15`, `min_samples_leaf=1`, `class_weight='balanced'`, `random_state=42`.

---

## 📈 Benchmark Results & Performance Summary

| Model Architecture | Hyperparameters | Test Accuracy | Macro Precision | Macro Recall | **Macro F1-Score** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** *(Selected)* | `max_depth=15, min_samples_leaf=1, balanced` | **99.69%** | **98.78%** | **98.94%** | **0.9886 (98.86%)** |
| **Multinomial Logistic Regression** | `C=2, class_weight='balanced'` | 99.44% | 98.06% | 98.28% | 0.9815 (98.15%) |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=7, metric='euclidean'` | 99.44% | 98.04% | 98.07% | 0.9805 (98.05%) |

---

## 🎯 Confusion Matrix Analysis Through a Business Lens

In facility operations, **not all misclassifications carry equal cost**:
* **Benign Error:** Misclassifying $2$ people as $3$ people has minimal impact; HVAC remains on and airflow adjusts slightly.
* **Critical Operational Error:** Misclassifying $0$ people as $3$ people (or $3$ people as $0$) is unacceptable. Predicting an empty room when three occupants are present causes automated systems to cut off climate control, exhaust fans, and lighting during an active meeting.

### Model Confusion Matrices (Test Set $N=1,621$)

#### 1. Decision Tree (Champion Model)
```
                Predicted 0   Predicted 1   Predicted 2   Predicted 3
Actual Class 0:    1317            0             0             0
Actual Class 1:       0           73             0             0
Actual Class 2:       0            1           116             3
Actual Class 3:       0            0             1           110
```
* **Critical Error Rate:** **0%**. Zero instances of Class 0 misclassified as Class 3 (or vice versa).
* **Minor Boundary Noise:** Only 5 total misclassifications across 1,621 records, solely occurring between adjacent occupancy states ($2 \leftrightarrow 3$ and $2 \leftrightarrow 1$).

#### 2. Logistic Regression
```
                Predicted 0   Predicted 1   Predicted 2   Predicted 3
Actual Class 0:    1316            0             0             1  <-- 1 false occupancy alarm
Actual Class 1:       0           73             0             0
Actual Class 2:       0            0           114             6
Actual Class 3:       0            0             2           109
```

#### 3. K-Nearest Neighbors (KNN)
```
                Predicted 0   Predicted 1   Predicted 2   Predicted 3
Actual Class 0:    1317            0             0             0
Actual Class 1:       0           73             0             0
Actual Class 2:       0            0           114             6
Actual Class 3:       0            0             3           108
```

---

## 💼 Operationalizing Predictions for Facilities Management

| Model Prediction | Operational Action Triggered | Business Value Delivered |
| :---: | :--- | :--- |
| **`0` Occupants** | • Release unused room reservations after a 10-minute grace window.<br>• Transition HVAC to energy-saver setback mode.<br>• Extinguish overhead illumination. | **Energy Conservation & Real Estate Optimization:** Eliminates ghost bookings and cuts utility overhead. |
| **`1` Occupant** | • Flag room as occupied for single-person privacy.<br>• Maintain baseline comfort temperature. | **Space Governance:** Detects 8-person conference rooms being monopolized by solo workers. |
| **`2` Occupants** | • Maintain standard ambient conditions.<br>• Track collaborative utilization trends. | **Space Planning:** Measures real-world 1-on-1 and peer collaboration density. |
| **`3` Occupants** | • Ramp up ventilation rates to flush $CO_2$.<br>• Trigger meeting room overflow alerts if capacity exceeds thresholds. | **Health, Wellness & Safety:** Prevents cognitive fatigue from $CO_2$ buildup and aids emergency accounting. |

---

## 🏛️ Production Readiness & Executive Recommendation

### Recommendation for Head of Facilities
> **Recommendation:** Proceed with the **Decision Tree Classifier** as the inference engine for the Phase 2 multi-room pilot.

* **Feasibility Proved:** The Proof of Concept clearly demonstrates that commodity IoT sensor streams (without invasive cameras or microphones) can accurately predict occupancy levels ($>98.8\%$ Macro F1).
* **Key Driving Signals:** $CO_2$ accumulation rate (`S5_CO2_Slope`), sound amplitude (`S1_Sound`), and directional illuminance are the most decisive indicators of active human count.
* **Interpretability:** Decision tree splits mirror physical logic (e.g., $CO_2 > \tau$ and Sound $> \theta$), making decisions auditable for facility operators.

### ⚠️ Pre-Deployment Prerequisites (Why Company-Wide Rollout Should Wait)
This PoC was trained on **one specific meeting room** under a controlled observational period. To transition from PoC to enterprise-grade production, the following real-world factors must be addressed:
1. **Room Geometry & Volume Disparities:** A 500 ppm $CO_2$ elevation in a small huddle room represents fewer occupants than in an executive boardroom.
2. **Sensor Placement Variations:** Sensors placed near supply vents, windows, or doors experience draft artifacts and sunlight washouts.
3. **Multi-Room Diversity Training:** Data must be collected across multiple room sizes, HVAC ventilation rates, and floor layouts prior to global deployment.

---

## 🗣️ The Interview Narrative (STAR Method)

When discussing this project with hiring managers and technical interviewers:

> *"At NovaSpace Technologies, the Facilities team faced a major blind spot: they knew when rooms were booked, but not when they were actually used. Meetings got cancelled, lights and AC ran in empty rooms, and large conference spaces were monopolized by single individuals.*
> 
> *As a Data Engineer, I was tasked with determining whether the telemetry our platform already collected—temperature, sound, light, CO₂, and PIR motion—could solve this as a 4-class occupancy estimation problem (0 to 3 people).*
> 
> *Before modeling, I conducted a data quality and engineering audit. I verified that raw Date/Time timestamps were transformed into cyclical hour and weekday features to prevent temporal data leakage. I also standardized features, which was vital since CO₂ readings in the thousands would otherwise distort distance calculations in algorithms like KNN.*
> 
> *Recognizing that the room is empty over 81% of the time, I evaluated Multinomial Logistic Regression, KNN, and Decision Trees using **Macro-averaged F1-score**, ensuring minority classes were not eclipsed. The Decision Tree emerged as the top performer with a 98.86% Macro F1-score and zero critical misclassifications between 0 and 3 occupants.*
> 
> *Crucially, I advised the Head of Facilities not to roll this out company-wide immediately. Because the PoC was calibrated on a single room's layout, scaling effectively requires training across multiple room volumes and HVAC profiles."*

---

## 📂 Repository Structure

```text
├── ML_Hackthon_project2.ipynb   # Complete Jupyter Notebook (EDA, Preprocessing, Modeling)
├── README.md                    # Project documentation & business report
└── predictions.csv              # Model inference predictions on holdout test set (2,026 records)
```

---

## 🛠️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/smart-room-occupancy-poc.git
   cd smart-room-occupancy-poc
   ```

2. **Install required dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. **Execute the analysis:**
   Launch Jupyter Notebook or execute the cells in `ML_Hackthon_project2.ipynb`:
   ```bash
   jupyter notebook ML_Hackthon_project2.ipynb
   ```
