"""
train_models.py - One-time script to train and save all ML model artifacts.
Run this script once before starting the Flask server.
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) in ("app", "gui") else BASE_DIR

MODELS_DIR = os.path.join(ROOT_DIR, 'models') if os.path.exists(os.path.join(ROOT_DIR, 'models')) else os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets') if os.path.exists(os.path.join(ROOT_DIR, 'datasets')) else os.path.join(BASE_DIR, '..', 'datasets')
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42

print("=" * 60)
print("ML CAPSTONE - MODEL TRAINING PIPELINE")
print("=" * 60)

# ============================================================
# 1. REGRESSION - Diamonds Dataset
# ============================================================
print("\n[1/3] Training Regression Models (Diamonds)...")

df_diamonds = pd.read_csv(os.path.join(DATASETS_DIR, 'diamonds.csv'))
print(f"  Loaded diamonds: {df_diamonds.shape}")

# Data cleaning (from notebook)
df_diamonds = df_diamonds.drop_duplicates()
df_diamonds = df_diamonds[(df_diamonds['x'] > 0) & (df_diamonds['y'] > 0) & (df_diamonds['z'] > 0)]
df_diamonds = df_diamonds[(df_diamonds['y'] <= 20) & (df_diamonds['z'] <= 20)]
df_cleaned_d = df_diamonds.copy()

# Feature engineering: volume
df_cleaned_d['volume'] = df_cleaned_d['x'] * df_cleaned_d['y'] * df_cleaned_d['z']

num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume']
cat_cols = ['cut', 'color', 'clarity']

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

X = df_cleaned_d[num_cols + cat_cols]
y = df_cleaned_d['price']

price_bins = pd.qcut(y, q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=price_bins)

preprocessor_reg = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OrdinalEncoder(categories=[cut_order, color_order, clarity_order]), cat_cols)
])

preprocessor_reg.fit(X_train)
X_train_proc = preprocessor_reg.transform(X_train)
X_test_proc = preprocessor_reg.transform(X_test)

def eval_reg(y_true, y_pred):
    return {
        'r2': round(float(r2_score(y_true, y_pred)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        'mae': round(float(mean_absolute_error(y_true, y_pred)), 2)
    }

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

reg_models = {
    'Random Forest':                  RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_split=5, random_state=SEED, n_jobs=-1),
    'Gradient Boosting':              GradientBoostingRegressor(random_state=SEED),
    'K-Nearest Neighbors':            KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'Decision Tree':                  DecisionTreeRegressor(random_state=SEED),
    'Polynomial Regression':          Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'Ridge Regression':               Ridge(),
    'Linear Regression':              LinearRegression(),
    'Lasso Regression':               Lasso(alpha=0.1, max_iter=2000, random_state=SEED),
    'ElasticNet':                     ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=SEED),
    'Support Vector Regressor (SVR)': SVR(C=1.0, epsilon=0.1, max_iter=3000),
}

reg_results = {}
best_reg_name = 'Random Forest'
best_reg_model = None

for name, model in reg_models.items():
    print(f"    Training {name}...")
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    metrics = eval_reg(y_test, y_pred)
    reg_results[name] = metrics
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    joblib.dump(model, os.path.join(MODELS_DIR, f'reg_{safe_name}.pkl'))
    print(f"      R2={metrics['r2']:.4f} RMSE=${metrics['rmse']:,.0f} MAE=${metrics['mae']:,.0f}")
    if name == best_reg_name:
        best_reg_model = model

# Save best model predictions for chart data
y_pred_best_reg = best_reg_model.predict(X_test_proc)
residuals = (y_test.values - y_pred_best_reg).tolist()

# Sample for scatter chart (max 500 points)
sample_idx = np.random.RandomState(SEED).choice(len(y_test), min(500, len(y_test)), replace=False)
actual_vs_pred = {
    'actual': y_test.values[sample_idx].tolist(),
    'predicted': y_pred_best_reg[sample_idx].tolist(),
    'residuals': np.array(residuals)[sample_idx].tolist()
}

# Feature importance for RF
fi_rf = best_reg_model.feature_importances_
feature_names_reg = num_cols + cat_cols
feature_importance_reg = sorted(
    [{'feature': f, 'importance': round(float(v), 4)} for f, v in zip(feature_names_reg, fi_rf)],
    key=lambda x: -x['importance']
)

# Save regression artifacts
joblib.dump(preprocessor_reg, os.path.join(MODELS_DIR, 'reg_preprocessor.pkl'))
joblib.dump(best_reg_model, os.path.join(MODELS_DIR, 'reg_best_model.pkl'))
joblib.dump(reg_models['Linear Regression'], os.path.join(MODELS_DIR, 'reg_linear.pkl'))
joblib.dump(reg_models['Ridge Regression'], os.path.join(MODELS_DIR, 'reg_ridge.pkl'))
joblib.dump(reg_models['Gradient Boosting'], os.path.join(MODELS_DIR, 'reg_gb.pkl'))

reg_data = {
    'models': reg_results,
    'best_model': best_reg_name,
    'feature_names': feature_names_reg,
    'feature_importance': feature_importance_reg,
    'actual_vs_pred': actual_vs_pred,
    'dataset_info': {
        'total_rows': len(df_diamonds),
        'clean_rows': len(df_cleaned_d),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'target': 'price',
        'price_min': round(float(y.min()), 2),
        'price_max': round(float(y.max()), 2),
        'price_mean': round(float(y.mean()), 2),
    },
    'categorical_info': {
        'cut_options': cut_order,
        'color_options': color_order,
        'clarity_options': clarity_order
    }
}

with open(os.path.join(MODELS_DIR, 'reg_data.json'), 'w') as f:
    json.dump(reg_data, f)
print(f"  Regression models trained and saved.")

# ============================================================
# 2. CLASSIFICATION - Fraud Detection Dataset
# ============================================================
print("\n[2/3] Training Classification Models (Fraud Detection)...")

df_fraud = pd.read_parquet(os.path.join(DATASETS_DIR, 'fraud_detection.parquet'))
print(f"  Loaded fraud dataset: {df_fraud.shape}")
print(f"  Columns: {list(df_fraud.columns)}")
print(f"  Target distribution:\n{df_fraud['risk_label'].value_counts()}")

df_clf = df_fraud.drop_duplicates().copy()

# Feature engineering
df_clf['amount_ratio'] = df_clf['transaction_amount'] / (df_clf['avg_transaction_amount_30d_customer'] + 1)

num_features = [
    'transaction_hour', 'transaction_day_of_week', 'account_age_days',
    'previous_chargebacks', 'customer_total_transactions_30d', 'customer_risk_score',
    'transaction_amount', 'avg_transaction_amount_30d_customer',
    'transaction_velocity_1h', 'transaction_velocity_24h', 'amount_ratio'
]

cat_features = [
    'merchant_category', 'transaction_country', 'device_type',
    'transaction_type', 'geo_location_region'
]

# Find binary features (those with only 0/1 values, excluding target and ids)
bin_features = []
exclude = {'risk_label', 'transaction_id', 'customer_id'} | set(num_features) | set(cat_features)
for col in df_clf.columns:
    if col not in exclude:
        unique_vals = set(df_clf[col].unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            bin_features.append(col)

print(f"  Numerical features ({len(num_features)}): {num_features}")
print(f"  Categorical features ({len(cat_features)}): {cat_features}")
print(f"  Binary features: {bin_features}")

predictor_cols = num_features + cat_features + bin_features

X_clf = df_clf[predictor_cols]
y_clf = df_clf['risk_label']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
)

preprocessor_clf = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ('bin', 'passthrough', bin_features)
])

preprocessor_clf.fit(X_train_c)
X_train_cp = preprocessor_clf.transform(X_train_c)
X_test_cp = preprocessor_clf.transform(X_test_c)

def eval_clf(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred, average='weighted')), 4),
        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4),
        'confusion_matrix': cm.tolist(),
        'roc_fpr': [round(v, 4) for v in fpr.tolist()[::10]],  # downsample
        'roc_tpr': [round(v, 4) for v in tpr.tolist()[::10]],
    }

clf_model_configs = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=SEED),
    'Support Vector Machine': SVC(kernel='rbf', C=10, class_weight='balanced', probability=True, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED),
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=30, random_state=SEED, n_jobs=-1),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001, learning_rate_init=0.001, max_iter=200, early_stopping=True, random_state=SEED),
}

clf_results = {}
best_clf_name = 'Random Forest'
best_clf_model = None

for name, model in clf_model_configs.items():
    print(f"    Training {name}...")
    model.fit(X_train_cp, y_train_c)
    y_pred = model.predict(X_test_cp)
    y_prob = model.predict_proba(X_test_cp)[:, 1]
    metrics = eval_clf(y_test_c, y_pred, y_prob)
    clf_results[name] = metrics
    print(f"      Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")
    if name == best_clf_name:
        best_clf_model = model

# Get feature names after preprocessing
ohe_feature_names = preprocessor_clf.named_transformers_['cat'].get_feature_names_out(cat_features)
all_feature_names = num_features + list(ohe_feature_names) + bin_features

# Feature importance for Random Forest
fi_rf_clf = best_clf_model.feature_importances_
feature_importance_clf = sorted(
    [{'feature': f.split('__')[-1], 'importance': round(float(v), 4)}
     for f, v in zip(all_feature_names, fi_rf_clf)],
    key=lambda x: -x['importance']
)[:15]

# Save classification artifacts
joblib.dump(preprocessor_clf, os.path.join(MODELS_DIR, 'clf_preprocessor.pkl'))
joblib.dump(best_clf_model, os.path.join(MODELS_DIR, 'clf_best_model.pkl'))

# Save all clf models
for name, model in clf_model_configs.items():
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, os.path.join(MODELS_DIR, f'clf_{safe_name}.pkl'))

clf_data = {
    'models': clf_results,
    'best_model': best_clf_name,
    'feature_names': num_features + cat_features + bin_features,
    'feature_importance': feature_importance_clf,
    'dataset_info': {
        'total_rows': len(df_fraud),
        'clean_rows': len(df_clf),
        'train_rows': len(X_train_c),
        'test_rows': len(X_test_c),
        'target': 'risk_label',
        'fraud_count': int((y_clf == 1).sum()),
        'legit_count': int((y_clf == 0).sum()),
        'fraud_pct': round(float((y_clf == 1).mean() * 100), 2),
    },
    'predictor_cols': predictor_cols,
    'num_features': num_features,
    'cat_features': cat_features,
    'bin_features': bin_features,
    'cat_options': {
        feat: sorted(df_clf[feat].unique().tolist())
        for feat in cat_features
    }
}

with open(os.path.join(MODELS_DIR, 'clf_data.json'), 'w') as f:
    json.dump(clf_data, f)
print(f"  Classification models trained and saved.")

# ============================================================
# 3. CLUSTERING - Amazon Products
# ============================================================
print("\n[3/3] Training Clustering Model (Amazon Products)...")

csv_path = os.path.join(DATASETS_DIR, 'amzon_sample_pod_cat.csv')
df_raw = pd.read_csv(csv_path)
print(f"  Loaded Amazon dataset: {df_raw.shape}")

df_clean = df_raw[df_raw['price'] > 0].copy().reset_index(drop=True)

K_FINAL = 4
df_clean['has_rating'] = (df_clean['stars'] > 0).astype(int)
valid_median_rating = float(df_clean[df_clean['stars'] > 0]['stars'].median())
df_clean['stars_imputed'] = df_clean['stars'].replace(0, valid_median_rating)
df_clean['has_discount'] = ((df_clean['listPrice'] > 0) & (df_clean['listPrice'] > df_clean['price'])).astype(int)
df_clean['log_price'] = np.log1p(df_clean['price'])
df_clean['log_reviews'] = np.log1p(df_clean['reviews'])
df_clean['log_bought'] = np.log1p(df_clean['boughtInLastMonth'])

features_clust = ['log_price', 'log_reviews', 'log_bought', 'stars_imputed', 'has_rating', 'has_discount']
scaler_clust = StandardScaler()
X_scaled = scaler_clust.fit_transform(df_clean[features_clust].values)

kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEED)
kmeans.fit(X_scaled)
df_clean['Cluster'] = kmeans.labels_

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
df_clean['pca_x'] = X_pca[:, 0]
df_clean['pca_y'] = X_pca[:, 1]

sil_score = float(silhouette_score(X_scaled, kmeans.labels_, sample_size=10000, random_state=SEED))
db_score = float(davies_bouldin_score(X_scaled, kmeans.labels_))
ch_score = float(calinski_harabasz_score(X_scaled, kmeans.labels_))

# Save artifacts
joblib.dump(scaler_clust, os.path.join(MODELS_DIR, 'clust_scaler.pkl'))
joblib.dump(kmeans, os.path.join(MODELS_DIR, 'clust_kmeans.pkl'))
joblib.dump(pca, os.path.join(MODELS_DIR, 'clust_pca.pkl'))

print(f"  Clustering done. Silhouette={sil_score:.4f} DB={db_score:.4f} CH={ch_score:.0f}")
print(f"  Artifacts saved to: {MODELS_DIR}")
print("\nAll models trained and saved successfully!")
print("=" * 60)
