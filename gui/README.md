# ML Capstone GUI Application · 23CSE301

An interactive, multi-track Machine Learning GUI application built with **Streamlit**, **Scikit-Learn**, and **Plotly** covering all three coursework tracks for the 23CSE301 ML Capstone Project.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 2. (Optional) Retrain Models
All models are already pre-trained and serialized in `gui/models/`. If you want to re-train them:
```bash
cd gui
python train_models.py
```

### 3. Launch the Streamlit GUI
```bash
cd gui
streamlit run streamlit_app.py
```
Or run the helper script:
```bash
./run_app.bat
```
The app will automatically open in your default browser at:  
👉 **`http://localhost:8501`**

---

## 🎯 Features Across All 3 Tracks

### Track 1: 💎 Regression (Diamonds Dataset)
- **Objective**: Predict diamond prices (`$`) based on carat weight, cut, color, clarity, and dimensions (`x`, `y`, `z`, depth, table).
- **Trained Models**:
  - Random Forest Regressor ($R^2 \approx 0.98$, Best Model)
  - Gradient Boosting Regressor
  - Ridge Regression
  - Linear Regression
- **Tabs & Capabilities**:
  1. **🎯 Predict Price**: Real-time interactive price estimation with confidence ranges and feature breakdown.
  2. **📊 Model Comparison**: Interactive comparison table and bar charts ($R^2$, RMSE, MAE).
  3. **📉 Visualisations**: Actual vs Predicted scatter plots, Residual distribution, and Random Forest feature importances.

---

### Track 2: 🛡️ Classification (Financial Fraud Detection)
- **Objective**: Detect fraudulent transactions given transaction type, amount, origin/destination balances.
- **Trained Models**:
  - Random Forest Classifier (Optimized F1/ROC-AUC, Best Model)
  - Logistic Regression
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - *(Evaluated across 10 distinct classifiers in training pipeline)*
- **Tabs & Capabilities**:
  1. **🛡️ Predict Fraud**: Live fraud probability meter, risk verdict (Legitimate vs. Fraudulent flag), and before/after account balance visualization.
  2. **📊 Model Comparison**: Metrics table comparing Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
  3. **📈 Visualisations**: Confusion Matrix heatmap, ROC Curve, and Top Predictive Feature Importances.

---

### Track 3: 🔵 Clustering (Amazon Product Catalog Segmentation)
- **Objective**: Unsupervised segmentation of the Amazon product catalog into distinct strategic clusters using K-Means and PCA.
- **Identified Segments**:
  - **Cluster 0**: *Standard Full-Price Catalog* (Core Evergreen)
  - **Cluster 1**: *High-Velocity & Reviewed* (Social Proof Star)
  - **Cluster 2**: *Promotional & Discounted* (Discount Driven)
  - **Cluster 3**: *Unrated / New / Dormant* (Cold Start)
- **Tabs & Capabilities**:
  1. **🎯 Predict Cluster**: Input new product metrics (Price, List Price, Stars, Reviews, Bought in Last Month) to find its target cluster, distance to cluster centroids, and confidence scores.
  2. **🧭 Segment Explorer**: Cluster breakdown cards, interactive 2D PCA projection scatter plot, feature distributions, and catalog share pie chart.
  3. **📉 Elbow & Silhouette**: Elbow curve (Inertia vs. $K$) and Silhouette Score analysis justifying the optimal $K=4$ selection.
  4. **📊 Catalog Insights**: Statistical metrics, feature means, and top categories per segment.

---

## 📂 Project Structure

```
gui/
├── models/                     # Serialized scikit-learn models & preprocessors (.pkl, .json)
│   ├── reg_best_model.pkl      # Random Forest regressor
│   ├── reg_preprocessor.pkl    # Robust diamond preprocessor
│   ├── clf_best_model.pkl      # Random Forest fraud classifier
│   ├── clf_preprocessor.pkl    # Fraud transaction preprocessor
│   ├── clust_kmeans.pkl        # K-Means clustering model (K=4)
│   ├── clust_pca.pkl           # PCA projection model (2 components)
│   └── clust_scaler.pkl        # Robust scaler for Amazon features
├── train_models.py             # Complete training pipeline for all 3 tracks
├── streamlit_app.py            # Streamlit dashboard implementation
├── run_app.bat                 # One-click Windows launch script
└── README.md                   # This documentation
```
