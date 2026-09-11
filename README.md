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

## Repository Structure
- `notebooks/`:
  - `regression.ipynb`: Self-contained, fully executed notebook covering Milestone 1 (Loading & Audit) and Milestone 2 (EDA visualisations & observations).
- `datasets/`:
  - `diamonds.csv`: Official regression dataset ($53,940 \times 10$).
- `AGENTS.md`: Repository workflow guidelines.
- `README.md`: Project documentation and milestone tracking.

