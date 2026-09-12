# Diamond Price Prediction Using Machine Learning Regression

## Project Overview
This repository contains the Machine Learning Capstone Regression project focused on predicting diamond market prices based on physical and gemological characteristics.

### Formulated Regression Task:
$$\text{Diamond Characteristics } (X) \longrightarrow \text{Predict Diamond Price } (y)$$

- **Target Variable ($y$):** `price` (Continuous price in USD, ranging from \$326 to \$18,823).
- **Predictors ($X$):** 9 attributes including physical weight (`carat`), proportions (`depth`, `table`), millimeter dimensions (`x`, `y`, `z`), and quality grades (`cut`, `color`, `clarity`).
- **Dataset:** UCI / StatLib Diamonds Dataset (`datasets/diamonds.csv`, 53,940 records).

---

## Milestone 1: Dataset Loading & Audit (Completed)
- **Data Ingestion:** Ingested the complete `datasets/diamonds.csv` ($53,940$ rows $\times$ $10$ columns).
- **Integrity Audit:**
  - **Missing Values:** Exactly **0 missing values (0.00%)** across all 10 columns.
  - **Duplicate Records:** Identified **146 exact duplicate rows (0.27%)** flagged for the cleaning milestone.
  - **Data Types:** 6 numerical float features (`carat`, `depth`, `table`, `x`, `y`, `z`), 3 categorical string features (`cut`, `color`, `clarity`), and 1 integer target (`price`).
- **Target Analysis:**
  - Mean: \$3,932.80, Median: \$2,401.00, Standard Deviation: \$3,989.44.
  - Skewness: $+1.6184$ (right-skewed distribution with a high-value tail up to \$18,823).
  - 11,602 unique price levels confirming continuous regression behavior.
- **Visual Distribution:** Rendered target histogram and KDE with mean/median markers.

---

## Milestone 2: Exploratory Data Analysis (Completed)
- **Feature Distributions:**
  - **Numerical:** Histograms show `carat` and `price` are strongly right-skewed; `depth` (mean $\approx 61.7\%$) and `table` (mean $\approx 57.5\%$) exhibit symmetric bell-shaped distributions; `x`, `y`, `z` show multimodal dimensional patterns.
  - **Categorical:** Count plots show `Ideal` ($39.95\%$) and `Premium` ($25.57\%$) dominate cut grades; colors `G`, `E`, `F` are most frequent; clarities `SI1`, `VS2`, and `SI2` make up over $63\%$ of stones.
- **Correlation Heatmap:**
  - `carat` has the strongest positive correlation with `price` ($r = 0.92$), followed by dimensions `x` ($0.88$), `y` ($0.87$), and `z` ($0.86$).
  - Substantial multi-collinearity exists between $x, y, z$ and $carat$ ($r > 0.95$).
  - Proportions `depth` ($r = -0.01$) and `table` ($r = 0.13$) have minimal linear correlation with price.
- **Feature vs. Target Relationships:**
  - **Carat vs. Price:** Upward non-linear curve showing exponential price increases as carat weight rises.
  - **Dimension X vs. Price:** Strong positive correlation; confirms data entry anomalies at $x = 0$.
- **Categorical Price Variations:**
  - Boxplots across `cut`, `color`, and `clarity` show variations in price, with apparent counter-intuitive medians explained by confounding with diamond carat size.
- **Outlier Inspection:**
  - Identified severe dimensional anomalies (e.g. $y > 30\text{ mm}$, $z > 30\text{ mm}$, and 20 records with $x, y, \text{ or } z = 0$) flagged for removal in the upcoming cleaning milestone.

---

## Milestone 3: Data Cleaning (Completed)
- **Duplicate Removal:** Checked and removed **146 exact duplicate rows**, reducing sample size from $53,940$ to $53,794$ to prevent data leakage between train and test sets.
- **Invalid Physical Dimensions:** Filtered out **19 records** with physically impossible zero dimensions ($x \le 0$, $y \le 0$, or $z \le 0$), leaving $53,775$ rows.
- **Typographical Errors:** Removed **3 extreme dimensional transposition errors** ($y > 20\text{ mm}$ or $z > 20\text{ mm}$), leaving **$53,772$ clean rows**.
- **Outlier Assessment (IQR Method):**
  - Evaluated statistical outliers across all numerical columns.
  - Justified keeping legitimate high-carat and high-price observations to preserve true market price variation without artificial truncation.
- **Missing Value Status:** Verified $0$ missing values across all columns; no imputation required.
- **In-Memory Pipeline:** Cleaned data is maintained directly in-memory as `df_cleaned` throughout the notebook pipeline. No intermediate dataset files are created, keeping `datasets/diamonds.csv` as the sole dataset file.

---

## Milestone 4: Train/Test Split and Preprocessing (Completed)
- **Train/Test Partitioning:**
  - 80:20 split with `random_state=42`.
  - Training set: **43,017 samples**; Testing set: **10,755 samples**.
  - Target stratification implemented using 5 quantile price bins to ensure balanced price distributions across both splits.
  - Stratification verified: Training median price (\$2,400.00) and test median price (\$2,407.00) match closely.
- **Categorical Feature Encoding:**
  - Applied `OrdinalEncoder` to `cut`, `color`, and `clarity`, matching official gemological grading hierarchies (`Fair` < `Good` < ... < `Ideal`; `J` < `I` < ... < `D`; `I1` < `SI2` < ... < `IF`).
- **Numerical Feature Scaling:**
  - Standardized all 6 numerical features (`carat`, `depth`, `table`, `x`, `y`, `z`) using `StandardScaler` to mean 0 and variance 1.
- **Zero-Leakage Preprocessing Architecture:**
  - `ColumnTransformer` fitted **strictly on training data (`X_train`)** and applied to transform both `X_train` and `X_test`.
  - Zero test set information leaked into the scaling or encoding parameters.
---

## Milestone 5: Feature Engineering (Completed)
- **Engineered Feature:** `volume = x * y * z` (diamond bounding volume in $\text{mm}^3$).
- **Domain Rationale:** Combines length ($x$), width ($y$), and depth ($z$) into a single physical measurement of 3D stone size, directly mirroring the physical mass-density relationship ($\text{Mass} = \text{Density} \times \text{Volume}$).
- **Feature Characteristics:**
  - Min volume: $31.71\text{ mm}^3$, Max volume: $790.13\text{ mm}^3$, Mean volume: $129.78\text{ mm}^3$.
  - Missing values: Exactly **0**.
  - Correlation with price: **$r = 0.9235$** (strong non-linear upward curve).
- **Zero Target Leakage:** Derived exclusively from physical dimensions ($x, y, z$) without using target `price`.
- **Integrated Pipeline:** Feature created on `df_cleaned` before train/test splitting and standardized alongside other numerical features, yielding **10 total preprocessed features** ($7$ numerical + $3$ categorical).

---

---

# Fraud Detection Using Machine Learning Classification

## Project Overview
This track focuses on detecting fraudulent financial transactions using customer, account, transaction, and behavioral features.

### Formulated Classification Task:
$$\text{Transaction & Customer Attributes } (X) \longrightarrow \text{Predict Fraud Status } (y \in \{0, 1\})$$

- **Target Variable ($y$):** `risk_label`
  - `0` = **Not Fraud** (Legitimate transaction)
  - `1` = **Fraud** (Fraudulent transaction)
- **Predictors ($X$):** 20 features covering customer history, transaction context, device, geography, and velocity.
- **Dataset:** `datasets/fraud_detection.parquet` (100,000 records).

---

## Milestone 7: Dataset Loading & Initial Audit (Completed)
- **Data Ingestion:** Loaded `datasets/fraud_detection.parquet` ($100,000$ rows $\times$ $21$ columns).
- **Integrity Audit:**
  - **Missing Values:** Exactly **0 missing values (0.00%)** across all 21 columns.
  - **Duplicate Records:** Exactly **0 duplicate rows (0.00%)**.
  - **Data Types:** 10 numerical features, 5 categorical features, 3 binary flag features, 2 identifier features, and 1 integer target (`risk_label`).
- **Target Analysis:**
  - **Not Fraud (Class 0):** $98,500$ transactions ($98.50\%$).
  - **Fraud (Class 1):** $1,500$ transactions ($1.50\%$).
  - **Imbalance Ratio:** $65.67 : 1$.
  - Target distribution visualized with count plot; noted that accuracy alone is insufficient for evaluation in extreme imbalance.
- **Feature Catalog:**
  - **Identifiers:** `transaction_id`, `customer_id` (flagged for future removal to prevent memorization).
  - **Numerical:** `transaction_hour`, `transaction_day_of_week`, `account_age_days`, `previous_chargebacks`, `customer_total_transactions_30d`, `customer_risk_score`, `transaction_amount`, `avg_transaction_amount_30d_customer`, `transaction_velocity_1h`, `transaction_velocity_24h`.
  - **Categorical:** `merchant_category`, `transaction_country`, `device_type`, `transaction_type`, `geo_location_region`.
  - **Binary Flags:** `is_international`, `is_high_risk_merchant_category`, `is_weekend`.
- **Notebook:** Created and executed `notebooks/classification.ipynb`.

---

## Repository Structure
- `notebooks/`:
  - `regression.ipynb`: Diamond price prediction regression track (Milestones 1–5).
  - `classification.ipynb`: Fraud detection classification track (Milestone 7).
- `datasets/`:
  - `diamonds.csv`: Diamond price regression dataset ($53,940 \times 10$).
  - `fraud_detection.parquet`: Fraud detection classification dataset ($100,000 \times 21$).
- `AGENTS.md`: Repository workflow guidelines.
- `README.md`: Project documentation and milestone tracking.






