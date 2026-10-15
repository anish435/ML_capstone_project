"""Notebook-aligned data preparation, model registry and persisted training."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, silhouette_score, davies_bouldin_score,
    calinski_harabasz_score)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
CACHE = ROOT / ".cache" / "dashboard"
SEED = 42

REGISTRY = {
    "regression": {
        "Linear Regression": (LinearRegression, False), "Ridge": (Ridge, False),
        "Decision Tree": (lambda: DecisionTreeRegressor(random_state=SEED), False),
        "Random Forest": (lambda: RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_split=5, random_state=SEED, n_jobs=-1), False),
        "Gradient Boosting": (lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=.15, max_depth=5, random_state=SEED), False),
        "Lasso": ("pending", True), "ElasticNet": ("pending", True), "Polynomial Regression": ("pending", True),
    },
    "classification": {
        "Logistic Regression": (lambda: LogisticRegression(class_weight="balanced", random_state=SEED), True),
        "K-Nearest Neighbors": (lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1), True),
        "Gaussian Naive Bayes": (GaussianNB, False),
        "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=SEED), False),
        "Random Forest": (lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=1), False),
        "Gradient Boosting": (lambda: GradientBoostingClassifier(random_state=SEED), False),
        "AdaBoost": (lambda: AdaBoostClassifier(random_state=SEED), False),
        "Bagging": ("pending", False),
        "Support Vector Machine": ("pending", True),
        "MLP Neural Network": ("pending", True),
    },
    "clustering": {"K-Means": (lambda: KMeans(n_clusters=4, n_init=10, random_state=SEED), True), "Agglomerative Ward": ("pending", True)},
}

REG_NUM = ["carat", "depth", "table", "x", "y", "z", "volume"]
REG_CAT = ["cut", "color", "clarity"]
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
CLF_NUM = ["transaction_hour", "transaction_day_of_week", "account_age_days", "previous_chargebacks", "customer_total_transactions_30d", "customer_risk_score", "transaction_amount", "avg_transaction_amount_30d_customer", "transaction_velocity_1h", "transaction_velocity_24h", "amount_ratio"]
CLF_CAT = ["merchant_category", "transaction_country", "device_type", "transaction_type", "geo_location_region"]
CLF_BIN = ["is_international", "is_high_risk_merchant_category", "is_weekend"]
CLF_FEATURES = CLF_NUM + CLF_CAT + CLF_BIN
CLUST_FEATURES = ["log_price", "stars_imputed", "has_rating", "log_reviews", "has_reviews", "log_bought", "has_list_price", "has_discount"]

def prepare_regression():
    df = pd.read_csv(DATA / "diamonds.csv").drop_duplicates()
    df = df[(df.x > 0) & (df.y > 0) & (df.z > 0)]
    df = df[(df.y < 20) & (df.z < 20)].reset_index(drop=True)
    df["volume"] = df.x * df.y * df.z
    X, y = df[REG_NUM + REG_CAT], df.price
    bins = pd.qcut(y, q=5, labels=False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=bins)
    prep = ColumnTransformer([("num", StandardScaler(), REG_NUM), ("cat", OrdinalEncoder(categories=[CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]), REG_CAT)])
    return df, X, y, Xtr, Xte, ytr, yte, prep

def prepare_classification():
    path = DATA / "fraud_detection.parquet"
    if not path.exists():
        raise FileNotFoundError("datasets/fraud_detection.parquet is missing")
    df = pd.read_parquet(path).drop_duplicates().copy()
    df["amount_ratio"] = df.transaction_amount / (df.avg_transaction_amount_30d_customer + 1)
    for c in CLF_BIN:
        if c not in df: df[c] = 0
    X, y = df[CLF_FEATURES], df.risk_label
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    prep = ColumnTransformer([("num", StandardScaler(), CLF_NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CLF_CAT), ("bin", "passthrough", CLF_BIN)])
    return df, X, y, Xtr, Xte, ytr, yte, prep

def prepare_clustering():
    df = pd.read_csv(DATA / "amzon_sample_pod_cat.csv")
    df = df[df.price > 0].copy().reset_index(drop=True)
    df["has_rating"] = (df.stars > 0).astype(int)
    df["stars_imputed"] = df.stars.mask(df.stars <= 0, df.loc[df.stars > 0, "stars"].median())
    df["log_price"] = np.log1p(df.price)
    df["log_reviews"] = np.log1p(df.reviews)
    df["has_reviews"] = (df.reviews > 0).astype(int)
    df["log_bought"] = np.log1p(df.boughtInLastMonth)
    df["has_list_price"] = (df.listPrice > 0).astype(int)
    df["has_discount"] = ((df.listPrice > df.price) & (df.listPrice > 0)).astype(int)
    prep = StandardScaler()
    return df, df[CLUST_FEATURES], prep

def artifact_path(track): return CACHE / f"{track}.joblib"

def train_track(track, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = artifact_path(track)
    if p.exists() and not force:
        try: return joblib.load(p)
        except Exception: pass
    started = time.perf_counter()
    if track == "regression":
        df, X, y, Xtr, Xte, ytr, yte, prep = prepare_regression()
        A, B = prep.fit_transform(Xtr), prep.transform(Xte)
        models, metrics, preds = {}, {}, {}
        for name, (factory, _) in REGISTRY[track].items():
            if factory == "pending": continue
            model = factory() if callable(factory) else factory()
            model.fit(A, ytr)
            pred = model.predict(B)
            models[name] = model
            preds[name] = pred
            metrics[name] = {"R2": float(r2_score(yte, pred)), "RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "MAE": float(mean_absolute_error(yte, pred))}
        best = "Random Forest"
        yhat = preds[best]
        sample = np.random.RandomState(SEED).choice(len(yte), min(2500, len(yte)), replace=False)
        payload = {"track": track, "df": df, "X": X, "y": y, "X_test": Xte, "y_test": yte, "preprocessor": prep, "models": models, "predictions": preds, "metrics": metrics, "best": best, "actual_sample": yte.iloc[sample].to_numpy(), "pred_sample": yhat[sample], "residuals": (yte.to_numpy()-yhat), "fit_seconds": time.perf_counter()-started}
    elif track == "classification":
        df, X, y, Xtr, Xte, ytr, yte, prep = prepare_classification()
        A, B = prep.fit_transform(Xtr), prep.transform(Xte)
        models, metrics, preds, probs, curves = {}, {}, {}, {}, {}
        for name, (factory, _) in REGISTRY[track].items():
            if factory == "pending": continue
            model = factory() if callable(factory) else factory()
            model.fit(A, ytr)
            # Bound temporary allocations for neighbour and probability predictions.
            pred_parts, prob_parts = [], []
            batch_size = 2500
            for start in range(0, len(B), batch_size):
                chunk = B[start:start+batch_size]
                pred_parts.append(model.predict(chunk))
                prob_parts.append(model.predict_proba(chunk)[:, 1])
            pred, prob = np.concatenate(pred_parts), np.concatenate(prob_parts)
            models[name], preds[name], probs[name] = model, pred, prob
            metrics[name] = {"Accuracy": float(accuracy_score(yte,pred)), "Precision": float(precision_score(yte,pred,zero_division=0)), "Recall": float(recall_score(yte,pred,zero_division=0)), "Weighted F1": float(f1_score(yte,pred,average="weighted")), "ROC-AUC": float(roc_auc_score(yte,prob))}
            curves[name] = roc_curve(yte, prob)
        payload = {"track":track,"df":df,"X":X,"y":y,"X_test":Xte,"y_test":yte,"preprocessor":prep,"models":models,"predictions":preds,"probabilities":probs,"metrics":metrics,"roc":curves,"confusion":{n:confusion_matrix(yte,p) for n,p in preds.items()},"best":"Random Forest","fit_seconds":time.perf_counter()-started}
    elif track == "clustering":
        df, X, scaler = prepare_clustering()
        Z = scaler.fit_transform(X)
        model = KMeans(n_clusters=4,n_init=10,random_state=SEED).fit(Z)
        labels = model.labels_
        sample_labels = labels
        scores = {"Silhouette": float(silhouette_score(Z, labels, sample_size=10000, random_state=SEED)), "Davies-Bouldin": float(davies_bouldin_score(Z,labels)), "Calinski-Harabasz": float(calinski_harabasz_score(Z,labels)), "Inertia": float(model.inertia_)}
        pca = PCA(n_components=2, random_state=SEED)
        coords = pca.fit_transform(Z)
        sample = np.random.RandomState(SEED).choice(len(Z), min(6000,len(Z)), replace=False)
        sweep = {}
        for k in range(2,9):
            km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Z)
            sweep[k]={"inertia":float(km.inertia_),"silhouette":float(silhouette_score(Z,km.labels_,sample_size=10000,random_state=SEED))}
        payload={"track":track,"df":df,"X":X,"scaled":Z,"scaler":scaler,"model":model,"labels":labels,"metrics":scores,"pca":pca,"coords":coords,"sample_idx":sample,"sweep":sweep,"fit_seconds":time.perf_counter()-started}
    else: raise ValueError(track)
    joblib.dump(payload,p,compress=3)
    return payload

def load_all(force=False): return {t: train_track(t,force=force) for t in REGISTRY}

def cache_summary():
    return {t: (artifact_path(t).exists()) for t in REGISTRY}
