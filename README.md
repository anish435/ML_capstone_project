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

## Repository Structure
- `notebooks/`:
  - `regression.ipynb`: Self-contained, executed notebook with dataset loading, audit tables, and target distribution plot.
- `datasets/`:
  - `diamonds.csv`: Official regression dataset ($53,940 \times 10$).
- `AGENTS.md`: Repository workflow guidelines.
- `README.md`: Project documentation and milestone tracking.
