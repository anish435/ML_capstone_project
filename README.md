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

## Milestone 6: Linear Regression (Completed)
- **Model:** Ordinary Least Squares `LinearRegression` (`sklearn.linear_model`).
- **Data Configuration:** Evaluated on the standardized, ordinal-encoded test split ($10,755$ samples) using 10 preprocessed features without modifying or re-splitting data.
- **Evaluation Performance:**
  - **$R^2$ Score:** **`0.9093`** (Explains $90.93\%$ of total price variance).
  - **Root Mean Squared Error (RMSE):** **`$1,217.74`**
  - **Mean Absolute Error (MAE):** **`$797.70`**
- **Domain Observation:** Strong baseline linearity established; non-linear residual patterns indicate potential for regularized or non-linear tree-based models to achieve higher precision.

---

## Ridge Regression (Completed)
- **Model:** L2-Regularized `Ridge` (`sklearn.linear_model`, default $\alpha=1.0$).
- **Data Configuration:** Evaluated on the identical standardized, ordinal-encoded test split ($10,755$ samples, $10$ features).
- **Evaluation Performance:**
  - **$R^2$ Score:** **`0.9093`** (Matches Linear Regression at $90.93\%$ variance explained).
  - **Root Mean Squared Error (RMSE):** **`$1,217.70`** (Slight numerical improvement over \$1,217.74).
  - **Mean Absolute Error (MAE):** **`$797.76`**
- **Domain Observation:** Ridge Regression performs almost identically to Linear Regression because the primary linear relationship between carat/size and price is already strong. The L2 penalty stabilizes the model against multicollinearity between interrelated geometric features (`carat`, `volume`, $x, y, z$).

---

## Lasso Regression (Completed)
- **Model:** L1-Regularized `Lasso` (`sklearn.linear_model`, `max_iter=2000`, `tol=0.01`, `random_state=42`).
- **Data Configuration:** Evaluated on the identical standardized, ordinal-encoded test split ($10,755$ samples, $10$ features).
- **Evaluation Performance:**
  - **$R^2$ Score:** **`0.9091`** (Explains $90.91\%$ of total price variance).
  - **Root Mean Squared Error (RMSE):** **`$1,219.03`**
  - **Mean Absolute Error (MAE):** **`$799.20`**
- **Domain Observation:** Lasso achieves consistent ~90.9% accuracy matching Linear and Ridge Regression. While L1 regularization has the ability to zero out redundant features, all 10 diamond attributes (especially carat, dimensions, cut, color, clarity, and volume) contribute meaningful explanatory power and are retained.

---

## ElasticNet Regression (Completed)
- **Model:** ElasticNet combining L1 & L2 penalties (`sklearn.linear_model`, default $\alpha=1.0$, $\text{l1\_ratio}=0.5$, `random_state=42`).
- **Data Configuration:** Evaluated on the identical standardized, ordinal-encoded test split ($10,755$ samples, $10$ features).
- **Evaluation Performance:**
  - **$R^2$ Score:** **`0.8647`** (Explains $86.47\%$ of total price variance).
  - **Root Mean Squared Error (RMSE):** **`$1,487.62`**
  - **Mean Absolute Error (MAE):** **`$1,008.65`**
- **Domain Observation:** Under default parameters ($\alpha=1.0, \text{l1\_ratio}=0.5$), the dual regularization applies stronger shrinkage compared to Ridge or Lasso alone, causing slight underfitting. In high-dimensional multicollinear settings, ElasticNet provides robust feature grouping; hyperparameter tuning of $\alpha$ allows it to match the ~91% baseline accuracy.

---

## Polynomial Regression (Completed)
- **Model:** Degree-2 `PolynomialFeatures` + `LinearRegression` (`sklearn.preprocessing`, `sklearn.linear_model`).
- **Data Configuration:** `PolynomialFeatures(degree=2)` fitted **strictly on `X_train_processed`** ($43,017$ samples $\rightarrow$ $66$ polynomial and interaction features) and transformed onto `X_test_processed` ($10,755$ samples $\rightarrow$ $66$ features) with zero data leakage.
- **Evaluation Performance:**
  - **$R^2$ Score:** **`0.9642`** (Explains $96.42\%$ of total price variance—major improvement over linear baseline).
  - **Root Mean Squared Error (RMSE):** **`$765.52`** (Reduced by $37.1\%$ from \$1,217.74).
  - **Mean Absolute Error (MAE):** **`$453.21`** (Reduced by $43.2\%$ from \$797.70).
- **Domain Observation:** Capturing second-order non-linear terms (such as $\text{carat}^2$ and dimension interactions) dramatically improves model fit. Because diamond price scales exponentially with carat size and stone volume, polynomial terms model the upward curvature accurately without requiring complex black-box architectures.

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

## Milestone 8: Exploratory Data Analysis (Completed)
- **Numerical Feature Distributions (2.1):**
  - Examined 7 key continuous variables: `transaction_amount`, `customer_risk_score`, `previous_chargebacks`, `customer_total_transactions_30d`, `transaction_velocity_1h`, `transaction_velocity_24h`, and `account_age_days`.
  - `transaction_amount` is heavily right-skewed with typical purchases under \$500 and a high-value tail up to \$15,000.
  - `previous_chargebacks` is heavily zero-inflated (nearly 90% of accounts have 0 chargebacks).
  - `customer_risk_score` displays a bimodal pattern with peaks near 0–1 and 3–4.
- **Categorical Feature Distributions (2.2):**
  - `merchant_category`: Evenly balanced across all 10 categories (~10,000 transactions each).
  - `transaction_country`: US accounts for 55.0% of transactions, followed by UK (12.0%), Canada (8.1%), Australia (6.9%), and India (6.0%).
  - `device_type`: Mobile dominates with 54.7%, followed by desktop (30.2%), tablet (10.0%), and wearable (5.1%).
  - `transaction_type`: POS (40.0%) and Online (40.0%) represent the bulk of activity, with ATM and MOTO at 10.0% each.
- **Target Variable Distribution (2.3):**
  - Confirmed 98,500 legitimate (98.50%) vs. 1,500 fraudulent (1.50%) transactions ($65.67 : 1$ imbalance).
  - Established that model evaluation will focus on Recall, Precision, F1-Score, and PR-AUC rather than misleading overall accuracy.
- **Correlation Analysis (2.4):**
  - `customer_risk_score` ($r = +0.36$) and `previous_chargebacks` ($r = +0.30$) have the highest linear association with fraud.
  - Transaction velocity features correlate positively with fraud ($r = +0.14$ for 1h, $r = +0.13$ for 24h) and share moderate collinearity with each other ($r = +0.44$).
- **Feature vs. Fraud Relationships (2.5 & 2.6):**
  - **Chargebacks:** Accounts with 0 prior chargebacks are overwhelmingly legitimate (89,876 legitimate vs. only 8 fraud cases); having $\ge 1$ previous chargebacks drastically increases fraud probability.
  - **Customer Risk Score:** Legitimate transactions have a low median score ($\approx 0.14$), whereas fraudulent transactions exhibit an elevated median score ($\approx 3.45$).
  - **Transaction Amount:** Fraudulent transactions exhibit higher average (\$1,701.37 vs. \$835.33) and median (\$897.69 vs. \$354.12) purchase amounts.
- **Categorical Fraud Prevalence (2.7):**
  - Identified high-risk merchant categories: `gambling` (2.77%), `luxury_goods` (2.76%), `online_services` (2.74%), and `travel` (2.72%).
  - Identified low-risk everyday categories: `fuel` (0.45%), `grocery` (0.46%), and `restaurant` (0.47%).
- **EDA Summary (2.8):**
  - Zero missing values and zero duplicate rows across the entire dataset.
  - Clear separation signals identified for downstream data cleaning, preprocessing, and model training.

---

## Milestone 9: Data Cleaning (Completed)
- **Missing Values Handling (3.1):**
  - Checked all 21 columns; confirmed exactly **0 missing values (0.00%)**.
  - No imputation required; strategy justified to prevent synthetic artifacts in financial transactions.
- **Duplicate Transactions Check (3.2):**
  - Verified exact duplicate records; identified **0 duplicate rows (0.00%)**.
  - All 100,000 unique records preserved; justified that duplicate transactions cause data leakage.
- **Domain Validity Verification (3.3):**
  - Executed 7 domain checks on numerical attributes (`transaction_amount <= 0`, `account_age_days < 0`, `customer_total_transactions_30d < 0`, `previous_chargebacks < 0`, `transaction_velocity_1h < 0`, `transaction_velocity_24h < 0`, `customer_risk_score < 0`).
  - Confirmed 0 invalid values across all checks; all values strictly within valid financial domains.
- **Outlier Analysis & Strategy (3.4):**
  - Analyzed statistical outliers via the IQR method across key features (`transaction_amount`, `customer_risk_score`, `account_age_days`, `transaction_velocity_1h`, `transaction_velocity_24h`).
  - Discovered that **1,496 out of 1,500 total fraud cases (99.73%)** fall in the statistical outlier region of `customer_risk_score`.
  - Discovered that high-value transactions and velocity bursts carry disproportionately high fraud rates (up to 11.18%).
  - **Decision:** Retained 100% of legitimate extreme values; 0 outliers removed. Trimming them would catastrophically eliminate the minority fraud signal.
- **In-Memory Dataset Construction (3.5 & 3.6):**
  - Maintained cleaned data in-memory as `df_cleaned` ($100,000$ rows $\times$ $21$ columns).
  - Raw dataset `datasets/fraud_detection.parquet` kept completely unmodified.
  - Comprehensive Cleaning Summary table embedded in the notebook.

---

## Milestone 10: Feature Engineering (Completed)
- **Engineered Feature:** `amount_ratio`
- **Mathematical Formulation:**
  $$\text{amount\_ratio} = \frac{\text{transaction\_amount}}{\text{avg\_transaction\_amount\_30d\_customer} + 1}$$
- **Domain Justification:**
  - Compares each transaction amount against the customer's personal 30-day spending baseline.
  - Provides a normalized measure of spending abnormality without referencing the target label `risk_label`.
  - Smoothing constant ($+1$) guarantees zero division-by-zero errors.
- **Statistical Validation:**
  - Minimum: `0.0067`, Maximum: `173.1206`, Mean: `1.3475`, Median: `0.7577`.
  - Exactly **0 missing values** and **0 infinite values**.
- **Feature vs. Target Discriminative Power:**
  - Fraudulent transactions exhibit nearly double the median ratio (**1.39** vs. **0.75**) and 75th percentile (**3.04** vs. **1.51**) compared to legitimate purchases.
- **Predictor Catalog Update:**
  - Numerical predictors expanded to 11 features (`num_cols_updated = num_cols + ['amount_ratio']`).
  - Total input predictors for downstream preprocessing: **19 features** (11 numerical + 5 categorical + 3 binary flags).

---

## Milestone 11: Encoding, Scaling & Train/Test Splitting (Completed)
- **Stratified Train/Test Partitioning:**
  - 80:20 split with `random_state=42` and `stratify=y`.
  - **Training Set:** 80,000 transactions (78,800 legitimate / 98.50%, 1,200 fraud / 1.50%).
  - **Testing Set:** 20,000 transactions (19,700 legitimate / 98.50%, 300 fraud / 1.50%).
  - Stratification perfectly preserved without artificial resampling or data distortion.
- **Feature Identification (19 Predictors):**
  - **Numerical (11):** 10 base features + engineered `amount_ratio`.
  - **Categorical (5):** `merchant_category`, `transaction_country`, `device_type`, `transaction_type`, `geo_location_region`.
  - **Binary Flags (3):** `is_international`, `is_high_risk_merchant_category`, `is_weekend`.
  - **Excluded Identifiers:** `transaction_id` and `customer_id` omitted from $X$ to prevent memorization shortcuts.
- **Categorical Feature Encoding:**
  - Applied `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted **only** on `X_train`.
  - Expanded 5 categorical variables into 33 binary indicator columns (`(80000, 33)` and `(20000, 33)`).
- **Numerical Feature Scaling:**
  - Applied `StandardScaler()` fitted **only** on `X_train`.
  - Standardized 11 numerical features to mean 0 and unit variance (`(80000, 11)` and `(20000, 11)`).
- **Final Preprocessed Pipeline:**
  - Unified `ColumnTransformer` fitted strictly on `X_train` to generate `X_train_processed` (`80,000 x 47`) and `X_test_processed` (`20,000 x 47`).
  - Zero missing values, matching feature dimensions across splits.
- **Data Leakage Audit:**
  - Full 6-point verification passed: splitting preceded transformation; encoders/scalers fitted exclusively on training data; test set only transformed; target `risk_label` strictly isolated; identifiers excluded.

---

## Repository Structure
- `notebooks/`:
  - `regression.ipynb`: Diamond price prediction regression track (Milestones 1–6: Loading, Audit, EDA, Cleaning, Preprocessing, Linear Regression, Ridge, Lasso, ElasticNet & Polynomial Regression).
  - `classification.ipynb`: Credit card transaction fraud detection track (Milestones 7–11: Loading, Audit, EDA, Cleaning, Feature Engineering & Preprocessing).
- `datasets/`:
  - `diamonds.csv`: Diamond price regression dataset ($53,940 \times 10$).
  - `fraud_detection.parquet`: Fraud detection classification dataset ($100,000 \times 21$).
- `AGENTS.md`: Repository workflow guidelines.
- `README.md`: Project documentation and milestone tracking.










