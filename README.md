# 23CSE301 Machine Learning Capstone

End-to-end machine-learning study for the 2026-27 `23CSE301 Machine Learning` capstone covering Continuous Valuation (Regression), Imbalanced Financial Decisioning (Classification), and E-Commerce Catalog Market Segmentation (Clustering Track).

## Team

| Team member | Registration / Roll number | Track Responsibilities |
|---|---|---|
| Darisa Nageswar | CB.SC.U4CSE24113 | Diamond Volume Feature Engineering, Fraud Dataset Audit & EDA, Classification Naive Bayes & MLP, Regression Ensemble Tuning, Clustering Preprocessing & Elbow Method |
| Kotikalapudi Anish | CB.SC.U4CSE24125 | Diamond Dataset Audit & EDA, Classification Feature Engineering & Gradient Boosting, Regression Baselines & Integration, Clustering PCA & t-SNE Pipeline Integration |
| Vennapusa Sunil Kumar Reddy | CB.SC.U4CSE24158 | Diamond Data Cleaning & Preprocessing, Fraud Data Cleaning & Scaling, SVM, Decision Tree & AdaBoost, Clustering Hierarchical Dendrograms & Linkage Analysis |

Implementation tasks were divided collaboratively among the team members, and all pipelines were implemented successfully with all members participating in review, benchmarking, and verification.

---

## Capstone Problem Tracks

### 1. Regression: diamond market price estimation
Predict the market price of cut diamonds (`price`, USD) based on physical mass (`carat`), proportions (`depth`, `table`), millimeter dimensions (`x`, `y`, `z`), and official gemological quality grades (`cut`, `color`, `clarity`). Diamond pricing exhibits strong non-linear curvature: while stone mass and spatial volume dictate baseline price thresholds, qualitative grade multipliers induce significant valuation jumps.

The study uses an 80:20 train/test partition ($43,017$ training samples, $10,755$ held-out test samples) stratified across 5 quantile price bins to ensure identical target distributions across splits.

Source: [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/resulcaliskan/diamonds). Dataset file: [`datasets/diamonds.csv`](datasets/diamonds.csv) ($53,940$ original records $\rightarrow$ $53,772$ clean records, 10 features).

### 2. Classification: credit card transaction fraud detection
Classify transactional risk status as legitimate (`0`) or fraudulent (`1`) using transaction details, customer account history, device characteristics, geography, and velocity metrics. Detecting transaction fraud presents an extreme statistical challenge due to severe class imbalance ($98,500$ legitimate transactions / $98.50\%$ vs. $1,500$ fraudulent transactions / $1.50\%$, an imbalance ratio of $65.67:1$).

The study uses an 80:20 stratified partition ($80,000$ training transactions, $20,000$ held-out test transactions) to preserve the exact $1.50\%$ fraud prevalence in both splits. Because a trivial majority-class baseline would attain $98.50\%$ accuracy while detecting zero frauds, model evaluation strictly prioritizes Recall, Precision, Weighted F1, and ROC-AUC over naive accuracy.

Source: [HuggingFace Lead-AI Fraud Detection Dataset v2](https://huggingface.co/datasets/arun-gharami/lead-ai-fraud-detection-dataset-v2/tree/main/data). Dataset file: [`datasets/fraud_detection.parquet`](datasets/fraud_detection.parquet) ($100,000$ records, $21$ columns $\rightarrow$ $47$ preprocessed features).

### 3. Clustering: e-commerce product market segmentation (Review 2 Track)
Discover natural behavioral market segments within e-commerce catalog listings using multidimensional product characteristics, customer social proof (ratings and review volume), retail pricing dynamics, discount structures, and purchasing velocity. E-commerce platforms manage millions of heterogeneous product listings spanning varied consumer categories; unsupervised segmentation reveals structural catalog tiers (e.g., promotional high-turnover items vs. premium low-velocity niche goods) to drive data-informed inventory replenishment and automated pricing strategies.

The pipeline implements both required algorithms: **K-Means Clustering** (with inertia elbow curve optimization) and **Agglomerative Hierarchical Clustering** (with dendrogram exploration and Ward/Complete/Average linkage comparisons).

Source: Amazon Product Catalog Benchmark. Dataset file: [`datasets/amzon_sample_pod_cat.csv`](datasets/amzon_sample_pod_cat.csv) ($50,000$ original records, $13$ columns $\rightarrow$ $47,921$ clean records, 8 engineered features).

---

## Repository Structure

```text
datasets/
  diamonds.csv             Raw diamond regression dataset (53,940 records x 10 features)
  fraud_detection.parquet  Raw transaction fraud classification dataset (100,000 records x 21 features)
  amzon_sample_pod_cat.csv Raw Amazon product clustering dataset (50,000 records x 13 columns)
notebooks/
  regression.ipynb         Review 1 regression track (10 algorithms, tuning, visualizations)
  classification.ipynb     Review 1 & 2 classification track (10 algorithms, tuning, evaluation)
  clustering.ipynb         Review 2 clustering track (K-Means, Agglomerative, PCA, t-SNE, profiling)
README.md                  Project overview, problem formulations, results, and setup instructions
requirements.txt           Pinned Python dependencies
```

---

## Environment Setup

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

### Required Python Dependencies
- `pandas==3.0.3` (DataFrame manipulation and tabular data ingestion)
- `numpy==2.4.6` (Vectorized numerical computations)
- `scikit-learn==1.9.0` (Preprocessing pipelines, clustering, classification, and regression estimators)
- `scipy==1.18.0` (Hierarchical linkage matrices, dendrogram generation, and distance metrics)
- `matplotlib==3.11.0` (Static data visualization, dendrogram rendering, and diagnostic plotting)
- `seaborn==0.13.2` (Statistical charts, KDE distributions, and correlation heatmaps)
- `pyarrow==24.0.0` (Apache Parquet columnar format serialization)
- `jupyter==1.1.1` (Interactive notebook development environment)
- `nbformat==5.10.4` & `nbclient==0.11.0` (Notebook execution engine)

---

## Dataset Provenance and Preprocessing Pipelines

### Regression Track: Diamonds Dataset
- **Ingestion & Integrity Audit:** Loaded $53,940$ rows $\times$ $10$ columns. Verified exactly **0 missing values (0.00%)**. Identified $146$ exact duplicate rows ($0.27\%$). Target `price` spans from \$326 to \$18,823 (mean \$3,932.80, median \$2,401.00, skewness $+1.6184$).
- **Data Cleaning:** Removed $146$ duplicates, $19$ records with physically impossible zero dimensions ($x \le 0, y \le 0, \text{ or } z \le 0$), and $3$ extreme typographical measurement errors ($y > 20\text{ mm}$ or $z > 20\text{ mm}$), yielding **$53,772$ clean records** in `df_cleaned`. Raw CSV is maintained unmodified.
- **Feature Engineering:** Constructed spatial volume $\text{volume} = x \times y \times z$ ($\text{mm}^3$) based on the physical mass-density relationship ($\text{Mass} = \text{Density} \times \text{Volume}$). Exhibits a correlation of **$r = 0.9235$** with price, outperforming individual dimensional metrics.
- **Preprocessing Pipeline:** `ColumnTransformer` fitted **strictly on `X_train`** ($43,017$ samples) and applied to `X_test` ($10,755$ samples). Applied `OrdinalEncoder` to `cut`, `color`, and `clarity` preserving official grading hierarchies; applied `StandardScaler` to all 7 continuous features (`carat`, `depth`, `table`, `x`, `y`, `z`, `volume`).

### Classification Track: Fraud Detection Dataset
- **Ingestion & Integrity Audit:** Loaded $100,000$ transactions $\times$ $21$ columns. Verified exactly **0 missing values** and **0 duplicate rows**. Target `risk_label` comprises $98,500$ legitimate transactions ($98.50\%$) and $1,500$ fraudulent transactions ($1.50\%$).
- **Data Cleaning:** Verified 7 financial domain business rules (`transaction_amount <= 0`, `account_age_days < 0`, etc.); verified 100% compliance across all $100,000$ rows. Outlier analysis via IQR revealed that **$99.73\%$ ($1,496$ out of $1,500$) of all fraud cases** reside in the statistical outlier tail of `customer_risk_score`. Zero outliers were removed to preserve the authentic minority signal. Cleaned data is held in-memory as `df_cleaned`.
- **Feature Engineering:** Constructed spending abnormality ratio:
  $$\text{amount\_ratio} = \frac{\text{transaction\_amount}}{\text{avg\_transaction\_amount\_30d\_customer} + 1}$$
  Captures abnormal deviations from the customer's personal 30-day baseline without label leakage. Fraudulent transactions exhibit nearly double the median ratio ($1.39$ vs. $0.75$).
- **Preprocessing Pipeline:** `ColumnTransformer` fitted **strictly on `X_train`** ($80,000$ samples). Identifier columns (`transaction_id`, `customer_id`) excluded. Applied `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to 5 categorical features ($33$ binary dummy columns) and `StandardScaler()` to 11 numerical features $\rightarrow$ **$47$ total preprocessed features**.

### Clustering Track: Amazon Product Segmentation Dataset
- **Ingestion & Integrity Audit:** Loaded $50,000$ rows $\times$ $13$ columns. Verified **0 missing values** across all columns and **0 duplicate rows**. Extreme right-tail skewness was diagnosed across raw numeric features: `reviews` ($+127.84$), `boughtInLastMonth` ($+15.63$), and `price` ($+5.74$).
- **Data Cleaning:**
  1. *Zero Prices Exclusion:* Identified $2,079$ records ($4.16\%$) with `price <= 0.0`. In commercial e-commerce, legitimate retail goods do not retail for zero dollars; these reflect scraping extraction failures, out-of-stock items, or unavailable listings. Removing them leaves **$47,921$ clean records** without distorting distance spaces.
  2. *Unrated Product Imputation:* $3,091$ products ($6.18\%$) have `stars == 0.0`. We construct an indicator `has_rating = (stars > 0).astype(int)` and impute `stars` with the positive ratings median ($4.40$), preserving the unrated signal without numerical corruption.
- **Feature Engineering & Log Transformations:**
  1. *Variance Stabilization:* Applied $f(x) = \ln(1 + x)$ (`np.log1p`) to `price` (skew dropped from $5.74 \rightarrow 0.28$), `reviews` ($127.84 \rightarrow 4.10$), and `boughtInLastMonth` ($15.63 \rightarrow 1.44$).
  2. *Commercial Structural Flags:*
     - `has_reviews = (reviews > 0).astype(int)`
     - `has_list_price = (listPrice > 0).astype(int)`
     - `has_discount = ((listPrice > price) & (listPrice > 0)).astype(int)`
- **Mandatory Leakage Prevention:** Identifiers and unstructured text (`Unnamed: 0`, `asin`, `title`, `imgUrl`, `productURL`) and near-constant `isBestSeller` ($99.67\%$ False) are excluded. Ground truth `category_id` and `category_name` are **strictly sequestered** and never used for fitting, linkage selection, or tuning. They are reserved exclusively for post-clustering interpretation.
- **Feature Scaling:** `StandardScaler` applied to the final 8 engineered features: `['log_price', 'stars_imputed', 'has_rating', 'log_reviews', 'has_reviews', 'log_bought', 'has_list_price', 'has_discount']`.

---

## Run the Notebooks

```bash
jupyter lab
```

Run each notebook from the first cell to the last with the working directory set to the repository root. All random operations use `random_state=42` where applicable.

To execute all three notebooks non-interactively and verify complete end-to-end execution:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/regression.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/classification.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/clustering.ipynb
```

---

## Review 2 Results: Clustering Track

### 1. Hierarchical Linkage Strategy Comparison ($N=2,000, k=4$)

To select the mathematically and structurally optimal linkage criterion for Agglomerative Clustering, we evaluated Ward, Complete, and Average linkages on an identical representative evaluation sample ($N=2,000$, `random_state=42`):

| Linkage Strategy | Selected k | Cluster Size Distribution | Silhouette Score (↑) | Davies–Bouldin Index (↓) | Calinski–Harabasz Index (↑) | Compute Time (s) | Verdict |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Ward** | **4** | **{0: 1178, 1: 124, 2: 578, 3: 120}** | **0.4562** | **0.8442** | **1,035.98** | 0.157s | **Best Linkage (Selected)** |
| **Complete** | 4 | {0: 1719, 1: 154, 2: 124, 3: 3} | 0.1487 | 1.1107 | 317.08 | 0.114s | Severe cluster chaining & low silhouette |
| **Average** | 4 | {0: 1863, 1: 124, 2: 10, 3: 3} | 0.4195 | 0.8376 | 269.95 | 0.157s | Degenerate outlier isolation (sizes 10, 3) |

- **Linkage Analysis:** Ward linkage minimizes within-cluster variance using Euclidean distance, producing balanced, highly coherent clusters. Complete and Average linkages suffer from chaining and outlier sensitivity, collapsing over $85\%$ of observations into a single monolithic cluster.

---

### 2. Primary Algorithm Benchmark Comparison (Identical Sample $N=2,000, k=4$)

For a fair, unbiased algorithm comparison, both K-Means and Agglomerative Clustering (Ward) were evaluated on the **exact same evaluation sample ($N=2,000$, $k=4$)**:

| Algorithm | Evaluation Sample (N) | Selected k | Silhouette Score (↑) | Davies–Bouldin Index (↓) | Calinski–Harabasz Index (↑) | Cluster Cohesion & Separation |
|---|:---:|:---:|:---:|:---:|:---:|:---|
| **K-Means (Sample Benchmark)** | 2,000 | 4 | **0.4562** | **0.8442** | **1,035.98** | High cohesion, spherical centroid convergence |
| **Agglomerative Clustering (Ward)** | 2,000 | 4 | **0.4562** | **0.8442** | **1,035.98** | Identical structural partition to K-Means |

*Note: Both algorithms achieve identical performance scores ($0.4562$ Silhouette, $0.8442$ Davies-Bouldin) because Ward's objective function (minimizing total within-cluster variance) mathematically mirrors the K-Means inertia criterion.*

---

### 3. Full-Dataset K-Means Deployment Summary ($N=47,921, k=4$)

The final K-Means model was fitted across all $47,921$ clean product records:

| Metric | Full Dataset K-Means Value | Operational Interpretation |
|---|:---:|:---|
| **Total Clean Dataset Size (N)** | $47,921$ products | Full catalog deployment |
| **Selected Number of Clusters (k)** | $4$ | Justified by Elbow Curve inflection and Silhouette peak |
| **Silhouette Score** | **0.4602** | Evaluated on large 5,000-row sample ($s > 0.45$ confirms solid separation) |
| **Davies–Bouldin Index** | **0.8425** | Sub-1.0 index indicates compact clusters with low dispersion |
| **Calinski–Harabasz Index** | **24,383.14** | High between-to-within variance ratio across 48k rows |
| **Final Inertia** | $151,733.60$ | Converged minimum sum of squared errors |

---

### 4. Discovered Market Segment Profiles (Full Catalog)

| Cluster # | Discovered Segment Persona | Catalog Share | Mean Price | Median Price | Mean Reviews | Mean Monthly Sales | Discount Rate | Distinctive Commercial Profile |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | **High-Engagement / Social Proof Leaders** | 6.24% ($2,992$) | $58.16 | $39.99 | **1,523.0** | **256.4 units** | 34.0% | Heavy review accumulation, high consumer sales velocity, active market validation. |
| **1** | **Standard Full-Price Core Catalogue** | **58.88%** ($28,218$) | $64.92 | $42.99 | 0.0 | 27.2 units | 0.0% | Core long-tail catalogue items sold at full MSRP with 0 discounts and steady baseline demand. |
| **2** | **Discounted / Promotional Deals** | 28.78% ($13,790$) | $58.54 | $39.99 | 0.0 | 59.9 units | **100.0%** | Actively discounted products (100% discount flag) driving elevated monthly turnover. |
| **3** | **Premium / Unrated New Listings** | 6.10% ($2,921$) | **$101.06** | **$59.99** | 0.0 | 1.0 units | 22.8% | High average ticket price ($101.06), 0% customer ratings, newly launched or low-velocity luxury items. |

---

### 5. Mandatory Visualisations Implemented in `clustering.ipynb`
1. **K-Means Elbow Curve & Silhouette Analysis:** Dual-axis plot across $k \in [2, 8]$ showing inertia trajectory inflection and silhouette peak at $k=4$.
2. **Hierarchical Dendrograms:** Side-by-side truncated dendrograms ($p=25$) for Ward, Complete, and Average linkages with horizontal cut lines.
3. **Dual PCA 2D Cluster Scatter Plots:** 2D projections for both K-Means and Agglomerative Clustering along PC1 ($28.37\%$ explained variance) and PC2 ($24.77\%$ explained variance), capturing **$53.14\%$ cumulative variance** with plotted centroids.
4. **t-SNE 2D Manifold Visualisation:** Non-linear t-SNE projection (`perplexity=30`, `max_iter=1000`) coloured by K-Means cluster labels showing non-linear topological separations.
5. **Post-Hoc Ground-Truth Category Heatmap:** Cross-tabulation of discovered clusters against `category_name`, demonstrating how product categories (e.g. Vacuums, Suitcases) align with behavioral market tiers.

---

## Prior Review Results: Classification & Regression

### Classification Comparison (10 Capstone Algorithms)

All models were evaluated on the identical held-out test partition ($20,000$ transactions, $47$ preprocessed features):

| Alg # | Model | Test Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | Frauds Caught (out of 300) | False Alarms | Status |
|:---:|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 7 | **AdaBoost Classifier** (`n_estimators`=150, `lr`=1.0) | **0.9931** | **0.8498** | 0.6600 | **0.9927** | **0.9971** | 198 | 35 | Completed (Review 2) |
| 8 | **Gradient Boosting Classifier** (`n_estimators`=150, `lr`=0.1, `depth`=4) | 0.9916 | 0.7619 | 0.6400 | 0.9912 | 0.9963 | 192 | 60 | Completed (Review 2) |
| 9 | **Bagging Classifier** (`n_estimators`=50, `features`=0.8, `depth`=15) | 0.9910 | 0.8235 | 0.5133 | 0.9900 | 0.9956 | 154 | **33** | Completed (Review 2) |
| 6 | **Random Forest Classifier** (`n_estimators`=200) | 0.9906 | 0.7222 | 0.6067 | 0.9902 | 0.9943 | 182 | 70 | Completed (Review 2) |
| 10 | **MLP Classifier (Neural Network)** (`hidden`=(100, 50), `tanh`, `lr`=0.01) | 0.9899 | 0.6612 | 0.6700 | 0.9899 | 0.9946 | 201 | 103 | Completed (Review 2) |
| 5 | **Support Vector Machine** ($C=10$, RBF) | 0.9894 | 0.6457 | 0.6500 | 0.9894 | 0.9917 | 195 | 107 | Completed (Review 1) |
| 2 | **K-Nearest Neighbors** ($k=5$) | 0.9870 | 0.7356 | 0.2133 | 0.9835 | 0.8704 | 64 | **23** | Completed (Review 1) |
| 4 | **Decision Tree** (`max_depth`=10) | 0.9689 | 0.3059 | 0.8433 | 0.9760 | 0.9150 | 253 | 574 | Completed (Review 1) |
| 1 | **Logistic Regression** (Balanced) | 0.9550 | 0.2477 | **0.9800** | 0.9679 | 0.9916 | **294** | 893 | Completed (Review 1) |
| 3 | **Gaussian Naive Bayes** | 0.9499 | 0.2225 | 0.9367 | 0.9647 | 0.9781 | 281 | 982 | Completed (Review 1) |

---

### Regression Comparison (10 Algorithms)

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

---

## Methodological Limitations

1. **Hierarchical Subsampling Limitation:** Full Agglomerative Hierarchical Clustering requires $O(N^2)$ space complexity. For $47,921$ products, the distance matrix consumes $\approx 18.37\text{ GB}$ RAM, exceeding standard workstation memory. While a representative stratified sample of $N=2,000$ reliably captures macro-cluster boundaries and linkage behaviors, micro-clusters with $<0.05\%$ catalog prevalence cannot be detected in a sample.
2. **E-Commerce Scraped Zero Artifacts:** Approximately $4.16\%$ of catalog items had recorded prices of zero. While filtering them avoids distorting distance matrices, these listings could represent promotional free giveaways or out-of-stock items in real production environments.
3. **Dimensionality Reduction Interpretability:** PCA captures $53.14\%$ of cumulative variance across 2 components. While 2D visualizations effectively depict cluster separation and centroids, 2D projections cannot convey the entire 8-dimensional geometric manifold.

---

## Responsible-Use Statement

The clustering and classification models in this repository are intended as decision-support systems for e-commerce inventory optimization, promotional targeting, and financial risk triage. Unsupervised cluster boundaries should guide exploratory business analytics and marketing strategies rather than automated, unreviewed catalog deletions or vendor penalties.

---

## AI-Assistance Disclosure

Generative AI tools were used for code scaffolding, documentation formatting, and LaTeX equation typesetting in strict compliance with Amrita Vishwa Vidyapeetham course guidelines. All data cleaning strategies, feature engineering logic, algorithm selections, empirical metric verifications, and market interpretations were verified and validated by the student engineering team.