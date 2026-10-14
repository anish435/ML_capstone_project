# 23CSE301 Machine Learning Capstone

End-to-end machine-learning study for the 2026-27 `23CSE301 Machine Learning` capstone covering Continuous Valuation (Regression) and Imbalanced Financial Decisioning (Classification).

## Team

| Team member | Registration / Roll number | Review 1 responsibility |
|---|---|---|
| Darisa Nageswar | CB.SC.U4CSE24113 | Diamond Volume Feature Engineering, Fraud Dataset Audit & EDA, Classification Naive Bayes, Regression Ensemble Tuning |
| Kotikalapudi Anish | CB.SC.U4CSE24125 | Diamond Dataset Audit & EDA, Classification Feature Engineering, Regression Baseline Models & Integration |
| Vennapusa Sunil Kumar Reddy | CB.SC.U4CSE24158 | Diamond Data Cleaning & Preprocessing, Fraud Data Cleaning & Scaling, SVM & Decision Tree Classification |

Implementation tasks were divided collaboratively among the team members, and the complete pipeline was implemented successfully with all members participating in review and verification.

## Review 1 problems

### Regression: diamond market price estimation

Predict the market price of cut diamonds (`price`, USD) based on physical mass (`carat`), proportions (`depth`, `table`), millimeter dimensions (`x`, `y`, `z`), and official gemological quality grades (`cut`, `color`, `clarity`). Diamond pricing exhibits strong non-linear curvature: while stone mass and spatial volume dictate baseline price thresholds, qualitative grade multipliers induce significant valuation jumps.

The study uses an 80:20 train/test partition ($43,017$ training samples, $10,755$ held-out test samples) stratified across 5 quantile price bins to ensure identical target distributions across splits.

Source: [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/resulcaliskan/diamonds). Dataset file: [`datasets/diamonds.csv`](datasets/diamonds.csv) ($53,940$ original records $\rightarrow$ $53,772$ clean records, 10 features).

### Classification: credit card transaction fraud detection

Classify transactional risk status as legitimate (`0`) or fraudulent (`1`) using transaction details, customer account history, device characteristics, geography, and velocity metrics. Detecting transaction fraud presents an extreme statistical challenge due to severe class imbalance ($98,500$ legitimate transactions / $98.50\%$ vs. $1,500$ fraudulent transactions / $1.50\%$, an imbalance ratio of $65.67:1$).

The study uses an 80:20 stratified partition ($80,000$ training transactions, $20,000$ held-out test transactions) to preserve the exact $1.50\%$ fraud prevalence in both splits. Because a trivial majority-class baseline would attain $98.50\%$ accuracy while detecting zero frauds, model evaluation strictly prioritizes Recall, Precision, Weighted F1, and ROC-AUC over naive accuracy.

Source: [HuggingFace Lead-AI Fraud Detection Dataset v2](https://huggingface.co/datasets/arun-gharami/lead-ai-fraud-detection-dataset-v2/tree/main/data). Dataset file: [`datasets/fraud_detection.parquet`](datasets/fraud_detection.parquet) ($100,000$ records, $21$ columns $\rightarrow$ $47$ preprocessed features).

## Current Review 1 progress

- **Phase 0:** Problem framing, target selection, leakage audit, and evaluation protocols defined for both tracks.
- **Phase 1:** Repository structure, verified dataset acquisition, integrity audits, and reproducible environment established.
- **Phase 2:** Dataset audit, exploratory data analysis (EDA), cleaning decisions, deterministic feature engineering (`volume` for diamonds; `amount_ratio` for fraud), fixed 80:20 splits, and leakage-safe preprocessing pipelines implemented in both notebooks.
- **Phase 3:** All ten regression algorithms and all five Classification Part-A algorithms implemented, tuned via 3-fold training cross-validation, evaluated on held-out test sets, and documented.

Both Review 1 notebooks have been executed from top to bottom with visible outputs and zero execution errors.

## Repository structure

```text
datasets/
  diamonds.csv             Raw diamond regression dataset (53,940 records x 10 features)
  fraud_detection.parquet  Raw transaction fraud classification dataset (100,000 records x 21 features)
notebooks/
  regression.ipynb         Review 1 regression track (10 algorithms, tuning, visualizations)
  classification.ipynb     Review 1 classification track (Part-A 5 algorithms, tuning, evaluation)
README.md                  Project overview, problem formulations, results, and setup instructions
requirements.txt           Pinned Python dependencies
```

## Environment setup

Python 3.10+ (tested on Python 3.12 / 3.13) is used for the reproducible environment. All dependencies are pinned in [`requirements.txt`](requirements.txt) to guarantee deterministic execution across platforms:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Required Python dependencies
- `pandas==3.0.3` (DataFrame manipulation and tabular data ingestion)
- `numpy==2.4.6` (Vectorized numerical computations)
- `scikit-learn==1.9.0` (Preprocessing pipelines, regression, and classification estimators)
- `matplotlib==3.11.0` (Static data visualization and diagnostic plotting)
- `seaborn==0.13.2` (Statistical charts, KDE distributions, and confusion matrices)
- `pyarrow==24.0.0` (Apache Parquet columnar format serialization)
- `jupyter==1.1.1` (Interactive notebook development environment)
- `nbformat==5.10.4` & `nbclient==0.11.0` (Notebook execution engine)

## Dataset provenance and preprocessing pipelines

### Regression track: diamonds dataset
- **Ingestion & Integrity Audit:** Loaded $53,940$ rows $\times$ $10$ columns. Verified exactly **0 missing values (0.00%)**. Identified $146$ exact duplicate rows ($0.27\%$). Target `price` spans from \$326 to \$18,823 (mean \$3,932.80, median \$2,401.00, skewness $+1.6184$).
- **Data Cleaning:** Removed $146$ duplicates, $19$ records with physically impossible zero dimensions ($x \le 0, y \le 0, \text{ or } z \le 0$), and $3$ extreme typographical measurement errors ($y > 20\text{ mm}$ or $z > 20\text{ mm}$), yielding **$53,772$ clean records** in `df_cleaned`. Raw CSV is maintained unmodified.
- **Feature Engineering:** Constructed spatial volume $\text{volume} = x \times y \times z$ ($\text{mm}^3$) based on the physical mass-density relationship ($\text{Mass} = \text{Density} \times \text{Volume}$). Exhibits a correlation of **$r = 0.9235$** with price, outperforming individual dimensional metrics.
- **Preprocessing Pipeline:** `ColumnTransformer` fitted **strictly on `X_train`** ($43,017$ samples) and applied to `X_test` ($10,755$ samples). Applied `OrdinalEncoder` to `cut`, `color`, and `clarity` preserving official grading hierarchies; applied `StandardScaler` to all 7 continuous features (`carat`, `depth`, `table`, `x`, `y`, `z`, `volume`).

### Classification track: fraud detection dataset
- **Ingestion & Integrity Audit:** Loaded $100,000$ transactions $\times$ $21$ columns. Verified exactly **0 missing values** and **0 duplicate rows**. Target `risk_label` comprises $98,500$ legitimate transactions ($98.50\%$) and $1,500$ fraudulent transactions ($1.50\%$).
- **Data Cleaning:** Verified 7 financial domain business rules (`transaction_amount <= 0`, `account_age_days < 0`, etc.); verified 100% compliance across all $100,000$ rows. Outlier analysis via IQR revealed that **$99.73\%$ ($1,496$ out of $1,500$) of all fraud cases** reside in the statistical outlier tail of `customer_risk_score`. Zero outliers were removed to preserve the authentic minority signal. Cleaned data is held in-memory as `df_cleaned`.
- **Feature Engineering:** Constructed spending abnormality ratio:
  $$\text{amount\_ratio} = \frac{\text{transaction\_amount}}{\text{avg\_transaction\_amount\_30d\_customer} + 1}$$
  Captures abnormal deviations from the customer's personal 30-day baseline without label leakage. Fraudulent transactions exhibit nearly double the median ratio ($1.39$ vs. $0.75$).
- **Preprocessing Pipeline:** `ColumnTransformer` fitted **strictly on `X_train`** ($80,000$ samples). Identifier columns (`transaction_id`, `customer_id`) excluded. Applied `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to 5 categorical features ($33$ binary dummy columns) and `StandardScaler()` to 11 numerical features $\rightarrow$ **$47$ total preprocessed features**.

## Run the notebooks

```bash
jupyter lab
```

Run each notebook from the first cell to the last with the working directory set to the repository root. All random operations use `random_state=42` where applicable.

To execute both notebooks non-interactively and verify complete end-to-end execution:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/regression.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/classification.ipynb
```

## Evaluation principles

- Every algorithm within a track uses the exact same held-out test split ($10,755$ samples for regression; $20,000$ samples for classification).
- Preprocessing pipelines (`ColumnTransformer`) are fitted strictly on training data only to guarantee zero data leakage.
- Hyperparameters are optimized strictly through training-set 3-fold cross-validation (`GridSearchCV`).
- Regression evaluates $R^2$, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) across all 10 models, with algorithms ranked by $R^2$.
- Classification evaluates Accuracy, Precision, Recall, Weighted F1, Confusion Matrix, and ROC-AUC for all 5 Part-A models.
- Due to extreme $1.50\%$ fraud imbalance, model evaluation emphasizes Recall (fraud capture rate), Precision (false alarm minimization), and ROC-AUC over overall accuracy.
- The held-out test partition is never accessed during feature selection, scaling calibration, or hyperparameter search.

## Review 1 results

### Regression comparison (10 algorithms)

All models were evaluated on the identical held-out test partition ($10,755$ samples, $10$ preprocessed features), ranked by $R^2$:

| Model | Test R2 | RMSE ($) | MAE ($) | Rank |
|---|---:|---:|---:|:---:|
| **Random Forest Regressor** | **0.9811** | **556.06** | **269.34** | **1** |
| **Gradient Boosting Regressor** | **0.9755** | **632.82** | **349.89** | **2** |
| **K-Nearest Neighbors Regressor** ($k=5$) | 0.9707 | 692.50 | 368.71 | 3 |
| **Decision Tree Regressor** | 0.9661 | 745.08 | 360.01 | 4 |
| **Polynomial Regression** ($d=2$) | 0.9642 | 765.52 | 453.21 | 5 |
| **Ridge Regression** ($\alpha=1.0$) | 0.9093 | 1,217.70 | 797.76 | 6 |
| **Linear Regression** (OLS) | 0.9093 | 1,217.74 | 797.70 | 7 |
| **Lasso Regression** ($\alpha=1.0$) | 0.9091 | 1,219.03 | 799.20 | 8 |
| **ElasticNet Regression** ($\alpha=1.0, \text{l1\_ratio}=0.5$) | 0.8647 | 1,487.62 | 1,008.65 | 9 |
| **Support Vector Regressor** (Default SVR) | 0.6015 | 2,552.78 | 1,220.11 | 10 |

#### Hyperparameter tuning & diagnostic analysis
- **Tuned Gradient Boosting:** Hyperparameter optimization via 3-fold training CV (`n_estimators=150, learning_rate=0.1, max_depth=5`, CV $R^2 = 0.9819$) significantly enhanced performance: test $R^2$ improved to **`0.9817`**, test RMSE dropped from \$632.82 to **`$546.70`** (a $13.6\%$ error reduction), and test MAE dropped to **`$284.00`**.
- **Tuned Random Forest:** Tuned with `n_estimators=150, max_depth=None, min_samples_split=5` (CV $R^2 = 0.9811$), achieving **`0.9810`** test $R^2$, **`$556.70`** RMSE, and the lowest overall test MAE at **`$269.44`**.
- **Diagnostic Visualisations:**
  - *Per-Model Diagnostic Plots:* Dedicated Predicted vs. Actual and Residual scatter plots are implemented for each of the 10 regression models, alongside Feature Importance horizontal bar plots for tree-based models (Decision Tree, Random Forest, Gradient Boosting).
  - *Best-Model (Random Forest) Diagnostic Plots:*
    - *Predicted vs. Actual Plot:* Points cluster tightly along the $45^\circ$ diagonal line ($\text{Predicted} = \text{Actual}$) with high fidelity below \$10,000.
    - *Residual Plot:* Residuals ($y_{\text{test}} - \hat{y}$) are symmetrically dispersed around zero with zero mean bias across all valuation brackets.
    - *Feature Importance Breakdown:* 3D spatial metrics (`volume` at $62.44\%$ and dimension `y` at $21.53\%$) account for **$83.97\%$** of total predictive importance. Gemological quality grades (`clarity` at $6.34\%$, `carat` at $5.23\%$, `color` at $3.13\%$) form the secondary price modifiers.

---

### Classification comparison (6 algorithms)

All models were evaluated on the identical held-out test partition ($20,000$ transactions, $47$ preprocessed features):

| Rank | Model | Test Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | Frauds Caught (out of 300) | False Alarms |
|:---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | **Random Forest Classifier** (`n_estimators`=200) | **0.9906** | 0.7222 | 0.6067 | **0.9902** | **0.9943** | 182 | **70** |
| 2 | **Support Vector Machine** ($C=10$, RBF) | 0.9894 | 0.6457 | 0.6500 | 0.9894 | 0.9917 | 195 | 107 |
| 3 | **K-Nearest Neighbors** ($k=5$) | 0.9870 | **0.7356** | 0.2133 | 0.9835 | 0.8704 | 64 | **23** |
| 4 | **Decision Tree** (`max_depth`=10) | 0.9689 | 0.3059 | 0.8433 | 0.9760 | 0.9150 | 253 | 574 |
| 5 | **Logistic Regression** (Balanced) | 0.9550 | 0.2477 | **0.9800** | 0.9679 | 0.9916 | **294** | 893 |
| 6 | **Gaussian Naive Bayes** | 0.9499 | 0.2225 | 0.9367 | 0.9647 | 0.9781 | 281 | 982 |

#### Hyperparameter tuning & trade-off analysis
- **Random Forest Classifier:** Ensembling 200 decorrelated trees via bagging and random feature subspaces achieves the highest test accuracy (**`0.9906`**), Weighted F1 (**`0.9902`**), and ROC-AUC (**`0.9943`**). Reduces false alarms down to just **70** (Precision **`0.7222`**) while capturing **182** fraudulent transactions. Feature importance reveals `previous_chargebacks` ($34.7\%$) and `customer_risk_score` ($31.6\%$) form over $66\%$ of total decision weight, followed by velocity metrics and engineered `amount_ratio`.
- **Support Vector Machine (SVC):** Parameter search across $C \in \{0.1, 1, 10\}$ and kernels identified **$C=10, \text{kernel}='rbf'$** as optimal ($98.94\%$ test accuracy). Delivers a strong enterprise trade-off: intercepts **$65.0\%$** of all frauds ($195/300$) with $107$ false alarms ($0.9917$ ROC-AUC).
- **K-Nearest Neighbors (KNN):** Neighborhood tuning identified **$k=5$** as optimal ($98.705\%$ accuracy). Maximizes Precision (**`0.7356`**, only $23$ false alarms), but suffers from low Recall (**`0.2133`**), missing $236$ out of $300$ fraudulent transactions due to neighborhood dilution by the dominant majority class.
- **Decision Tree Classifier:** Depth optimization identified **`max_depth=10`** as optimal ($96.89\%$ test accuracy). Top split visualization confirms `customer_risk_score` and `previous_chargebacks` form the root decision boundaries, detecting **$253$** fraud cases (Recall: `0.8433`).
- **Logistic Regression & Naive Bayes:** Inversely weighted Logistic Regression and Gaussian Naive Bayes maximize Recall (**`0.9800`** and **`0.9367`**), detecting $294$ and $281$ frauds, but generate high false alarm rates ($893$ and $982$ false positives), making them suitable for high-sensitivity triage rather than automated blocking.

## Review 1 conclusions and limitations

- Diamond market valuation is fundamentally non-linear; tree-based ensembles (Random Forest, Gradient Boosting) outperform baseline linear formulations by over 7 percentage points in $R^2$ and reduce MAE by nearly two-thirds.
- 3D physical volume ($\text{mm}^3$) accounts for over $62\%$ of predictive power in diamond appraisal, confirming that physical mass and spatial dimensions dictate baseline pricing, while color, clarity, and cut act as secondary value adjustments.
- In financial fraud detection, overall accuracy is statistically misleading in severe class imbalance ($1.50\%$ positive class); model selection must be guided by operational cost trade-offs between direct financial losses from undetected frauds (Recall) versus customer friction from false positives (Precision).
- Support Vector Machine provides the most balanced enterprise profile ($65.0\%$ Recall, $64.6\%$ Precision, $0.9917$ ROC-AUC), whereas Logistic Regression serves as an optimal high-recall safety filter.
- Models reflect statistical associations from historical data and should serve as decision-support soft sensors rather than autonomous arbiters.

## Responsible-use statement

The fraud detection models developed in this study are intended exclusively as decision-support systems for transaction triage and risk prioritization, not for automated, unreviewed account blocking or financial denial. High-risk predictions should prompt stepped-up authentication (e.g., multi-factor verification) or human-analyst review. Decision thresholds must be continuously calibrated to minimize disparate impact across consumer segments and commercial merchant categories.

## AI-assistance disclosure

AI tools were used for assistance during the development of this project, mainly for coding, documentation, and general guidance.
They were also used to help with understanding concepts and structuring parts of the implementation.
The work was reviewed, tested, and modified by our team according to the project requirements.