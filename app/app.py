"""
ML Capstone - Streamlit Application
Three tracks: Regression (Diamonds), Classification (Fraud), Clustering (Amazon)
Run with: streamlit run streamlit_app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ML Capstone · 23CSE301",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) in ("app", "gui") else BASE_DIR

MODELS_DIR  = os.path.join(ROOT_DIR, "models") if os.path.exists(os.path.join(ROOT_DIR, "models")) else os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets") if os.path.exists(os.path.join(ROOT_DIR, "datasets")) else os.path.join(BASE_DIR, "..", "datasets")


# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

/* Typography & Base Variables */
:root {
    --bg-page: #FFFFFF;
    --bg-secondary: #FAF9F5;
    --accent: #D97757;
    --accent-hover: #C15F3C;
    --text-primary: #1F1E1D;
    --text-muted: #6B6A68;
    --border-color: #E8E6DC;
    --radius: 8px;
}

html, body, [class*="css"], .stMarkdown, p, span, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #1F1E1D !important;
}

/* Base Canvas & Background - Clean Crisp White */
.stApp {
    background-color: #FFFFFF !important;
    color: #1F1E1D !important;
}

/* Headings - Editorial Anthropic / Claude Aesthetic */
h1, .main-title {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
    color: #1F1E1D !important;
    letter-spacing: -0.015em !important;
    font-size: 2.1rem !important;
    margin-bottom: 0.4rem !important;
}

h2, h3 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
    color: #1F1E1D !important;
    letter-spacing: -0.01em !important;
}

h4, h5, h6 {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 600 !important;
    color: #1F1E1D !important;
    letter-spacing: -0.005em !important;
}

/* Sidebar - Subtle Warm Grey Canvas */
[data-testid="stSidebar"] {
    background-color: #FAF9F5 !important;
    border-right: 1px solid #E8E6DC !important;
}
[data-testid="stSidebar"] * {
    color: #1F1E1D !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1F1E1D !important;
}

/* Radio Navigation in Sidebar */
[data-testid="stSidebar"] .stRadio label { 
    background: #FFFFFF !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    display: block !important;
    cursor: pointer !important;
    border: 1px solid #E8E6DC !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    color: #1F1E1D !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stRadio label:hover { 
    border-color: #D97757 !important;
    background: #FAF9F5 !important;
    color: #1F1E1D !important; 
}

/* Section Kickers */
.kicker {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #D97757;
    margin-bottom: 4px;
}

/* Metric Cards - Minimalist Flat Bordered KPI Blocks */
.metric-card {
    background: #FAF9F5;
    border: 1px solid #E8E6DC;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: left;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1F1E1D;
    letter-spacing: -0.02em;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #6B6A68;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* Streamlit Native Metric Overrides */
[data-testid="stMetric"] {
    background-color: #FAF9F5 !important;
    border: 1px solid #E8E6DC !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    box-shadow: none !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: #6B6A68 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    color: #1F1E1D !important;
    letter-spacing: -0.02em !important;
}

/* Prediction Result Box - Flat Solid Alert Styling */
.pred-box {
    border-radius: 8px;
    padding: 20px;
    margin: 14px 0;
    border: 1px solid #E8E6DC;
    background: #FAF9F5;
}
.pred-box.success {
    background: #FAF9F5;
    border: 1px solid #C8E6C9;
    border-left: 4px solid #5B8266;
}
.pred-box.danger {
    background: #FAF9F5;
    border: 1px solid #F8D7DA;
    border-left: 4px solid #C15F3C;
}
.pred-box.neutral {
    background: #FAF9F5;
    border: 1px solid #E8E6DC;
    border-left: 4px solid #D97757;
}

.pred-label {
    font-size: 1.65rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: #1F1E1D;
    letter-spacing: -0.02em;
}
.pred-sub {
    font-size: 0.85rem;
    color: #6B6A68;
}

/* Cluster Badge */
.cluster-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 0.75rem;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid currentColor;
    background: #FFFFFF;
}

/* Info Table */
.info-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}
.info-table td {
    padding: 9px 12px;
    border-bottom: 1px solid #E8E6DC;
    font-size: 0.84rem;
}
.info-table td:first-child {
    color: #6B6A68;
    width: 55%;
}
.info-table td:last-child {
    font-weight: 600;
    color: #1F1E1D;
    text-align: right;
}

/* Section Header */
.section-hdr {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1F1E1D;
    margin: 18px 0 10px;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #E8E6DC;
    margin: 20px 0;
}

/* Cluster Profile Card */
.clu-card {
    background: #FAF9F5;
    border-radius: 8px;
    padding: 18px;
    border: 1px solid #E8E6DC;
    margin-bottom: 12px;
}

/* Distance Bar */
.dist-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0;
    font-size: 0.82rem;
}
.dist-name {
    width: 200px;
    color: #1F1E1D;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
}
.dist-bar-wrap {
    flex: 1;
    background: #E8E6DC;
    border-radius: 4px;
    height: 6px;
}
.dist-bar {
    height: 6px;
    border-radius: 4px;
}
.dist-val {
    width: 55px;
    text-align: right;
    color: #1F1E1D;
    font-weight: 600;
}

/* Confidence Bar */
.conf-wrap {
    background: #E8E6DC;
    border-radius: 4px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}
.conf-fill {
    height: 8px;
    border-radius: 4px;
}

/* Form Controls - Inputs & Dropdowns */
.stNumberInput input, .stSelectbox [data-baseweb="select"], .stTextInput input, .stTextArea textarea {
    background-color: #FAF9F5 !important;
    color: #1F1E1D !important;
    border: 1px solid #E8E6DC !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    box-shadow: none !important;
}
.stNumberInput input:focus, .stSelectbox [data-baseweb="select"]:focus, .stTextInput input:focus {
    border-color: #D97757 !important;
    box-shadow: 0 0 0 1px #D97757 !important;
    outline: none !important;
}
.stSelectbox [data-baseweb="select"] * {
    color: #1F1E1D !important;
}

/* Primary Action Buttons - Solid Orange Terracotta */
button[kind="primary"] {
    background-color: #D97757 !important;
    background-image: none !important;
    border: 1px solid #D97757 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 18px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
button[kind="primary"]:hover {
    background-color: #C15F3C !important;
    border-color: #C15F3C !important;
    box-shadow: none !important;
    transform: none !important;
}
button[kind="primary"] * {
    color: #FFFFFF !important;
}

/* Secondary Buttons */
button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    background-image: none !important;
    border: 1px solid #E8E6DC !important;
    color: #1F1E1D !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 18px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
    background-color: #FAF9F5 !important;
    border-color: #D5D2C7 !important;
    color: #1F1E1D !important;
}

/* Tabs - Clean Segmented Control with Active Orange Hairline */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #E8E6DC;
    padding: 0;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.88rem;
    color: #6B6A68 !important;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    transition: color 0.15s ease, border-color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1F1E1D !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #1F1E1D !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #D97757 !important;
    box-shadow: none !important;
}

/* Alerts / Banners */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid #E8E6DC !important;
    background-color: #FAF9F5 !important;
    color: #1F1E1D !important;
    box-shadow: none !important;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E8E6DC !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Hide Streamlit default branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────

def load_json_metadata(filename):
    """Load JSON metadata file reliably with UTF-8 encoding from MODELS_DIR."""
    target_path = os.path.join(MODELS_DIR, filename)
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading ML model artifacts…")
def load_models(_cache_key="20261016_v3_multimodel_live"):
    result = {"reg": {}, "clf": {}, "clust": {}}

    # ── Regression ──
    try:
        result["reg"]["preprocessor"] = joblib.load(os.path.join(MODELS_DIR, "reg_preprocessor.pkl"))
        result["reg"]["models"] = {
            "Random Forest":                  joblib.load(os.path.join(MODELS_DIR, "reg_best_model.pkl")),
            "Gradient Boosting":              joblib.load(os.path.join(MODELS_DIR, "reg_gradient_boosting.pkl")),
            "K-Nearest Neighbors":            joblib.load(os.path.join(MODELS_DIR, "reg_k_nearest_neighbors.pkl")),
            "Decision Tree":                  joblib.load(os.path.join(MODELS_DIR, "reg_decision_tree.pkl")),
            "Polynomial Regression":          joblib.load(os.path.join(MODELS_DIR, "reg_polynomial_regression.pkl")),
            "Ridge Regression":               joblib.load(os.path.join(MODELS_DIR, "reg_ridge_regression.pkl")),
            "Linear Regression":              joblib.load(os.path.join(MODELS_DIR, "reg_linear_regression.pkl")),
            "Lasso Regression":               joblib.load(os.path.join(MODELS_DIR, "reg_lasso_regression.pkl")),
            "ElasticNet":                     joblib.load(os.path.join(MODELS_DIR, "reg_elasticnet.pkl")),
            "Support Vector Regressor (SVR)": joblib.load(os.path.join(MODELS_DIR, "reg_support_vector_regressor_svr.pkl")),
        }
        result["reg"]["data"] = load_json_metadata("reg_data.json")
        result["reg"]["ready"] = True
    except Exception as e:
        result["reg"]["ready"] = False
        result["reg"]["error"] = str(e)

    # ── Classification ──
    try:
        result["clf"]["preprocessor"] = joblib.load(os.path.join(MODELS_DIR, "clf_preprocessor.pkl"))
        result["clf"]["models"] = {
            "Random Forest":          joblib.load(os.path.join(MODELS_DIR, "clf_best_model.pkl")),
            "Gradient Boosting":      joblib.load(os.path.join(MODELS_DIR, "clf_gradient_boosting.pkl")),
            "Logistic Regression":    joblib.load(os.path.join(MODELS_DIR, "clf_logistic_regression.pkl")),
            "AdaBoost":               joblib.load(os.path.join(MODELS_DIR, "clf_adaboost.pkl")),
            "Decision Tree":          joblib.load(os.path.join(MODELS_DIR, "clf_decision_tree.pkl")),
            "K-Nearest Neighbors":    joblib.load(os.path.join(MODELS_DIR, "clf_k-nearest_neighbors.pkl")),
            "Gaussian Naive Bayes":   joblib.load(os.path.join(MODELS_DIR, "clf_gaussian_naive_bayes.pkl")),
            "Support Vector Machine": joblib.load(os.path.join(MODELS_DIR, "clf_support_vector_machine.pkl")),
            "Bagging":                joblib.load(os.path.join(MODELS_DIR, "clf_bagging.pkl")),
            "MLP Neural Network":     joblib.load(os.path.join(MODELS_DIR, "clf_mlp_neural_network.pkl")),
        }
        result["clf"]["data"] = load_json_metadata("clf_data.json")
        result["clf"]["ready"] = True
    except Exception as e:
        result["clf"]["ready"] = False
        result["clf"]["error"] = str(e)

    # ── Clustering ──
    try:
        result["clust"]["scaler"] = joblib.load(os.path.join(MODELS_DIR, "clust_scaler.pkl"))
        result["clust"]["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "clust_kmeans.pkl"))
        result["clust"]["pca"]    = joblib.load(os.path.join(MODELS_DIR, "clust_pca.pkl"))
        result["clust"]["ready"] = True
    except Exception as e:
        result["clust"]["ready"] = False
        result["clust"]["error"] = str(e)

    return result


@st.cache_data(show_spinner="Preparing clustering data…")
def load_clustering_data():
    """Pre-compute all cluster analytics. Cached on first call."""
    SEED = 42
    K = 4

    df_raw = pd.read_csv(os.path.join(DATASETS_DIR, "amzon_sample_pod_cat.csv"))
    df = df_raw[df_raw["price"] > 0].copy().reset_index(drop=True)

    df["has_rating"]    = (df["stars"] > 0).astype(int)
    valid_median        = float(df[df["stars"] > 0]["stars"].median())
    df["stars_imputed"] = df["stars"].replace(0, valid_median)
    df["has_discount"]  = ((df["listPrice"] > 0) & (df["listPrice"] > df["price"])).astype(int)
    df["log_price"]     = np.log1p(df["price"])
    df["log_reviews"]   = np.log1p(df["reviews"])
    df["log_bought"]    = np.log1p(df["boughtInLastMonth"])

    features = ["log_price", "log_reviews", "log_bought", "stars_imputed", "has_rating", "has_discount"]

    M = load_models()
    X_scaled = M["clust"]["scaler"].transform(df[features].values)
    df["Cluster"] = M["clust"]["kmeans"].predict(X_scaled)
    X_pca = M["clust"]["pca"].transform(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    CLUSTER_META = {
        0: {"name": "Standard Full-Price Catalog", "tag": "Core Evergreen",    "color": "#D97757"},
        1: {"name": "High-Velocity & Reviewed",    "tag": "Social Proof Star", "color": "#5A6B7C"},
        2: {"name": "Promotional & Discounted",    "tag": "Discount Driven",   "color": "#5B8266"},
        3: {"name": "Unrated / New / Dormant",     "tag": "Cold Start",        "color": "#877B70"},
    }

    profiles = {}
    for c in range(K):
        c_df = df[df["Cluster"] == c]
        profiles[c] = {
            "id": c,
            "meta": CLUSTER_META[c],
            "count": len(c_df),
            "share": round(len(c_df) / len(df) * 100, 1),
            "mean_price": round(c_df["price"].mean(), 2),
            "median_price": round(c_df["price"].median(), 2),
            "mean_stars": round(c_df["stars_imputed"].mean(), 2),
            "mean_reviews": round(c_df["reviews"].mean(), 0),
            "mean_bought": round(c_df["boughtInLastMonth"].mean(), 0),
            "discount_rate": round(c_df["has_discount"].mean() * 100, 1),
            "rated_rate": round(c_df["has_rating"].mean() * 100, 1),
            "top_categories": c_df["category_name"].value_counts().head(5).to_dict(),
        }

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    sil = round(float(silhouette_score(X_scaled, df["Cluster"], sample_size=10000, random_state=SEED)), 4)
    db  = round(float(davies_bouldin_score(X_scaled, df["Cluster"])), 4)
    ch  = round(float(calinski_harabasz_score(X_scaled, df["Cluster"])), 1)

    elbow = [
        {"k": 2, "inertia": 230021.3, "silhouette": 0.3053, "davies_bouldin": 1.6180},
        {"k": 3, "inertia": 182768.0, "silhouette": 0.3612, "davies_bouldin": 1.2608},
        {"k": 4, "inertia": 144718.8, "silhouette": 0.3508, "davies_bouldin": 1.0759},
        {"k": 5, "inertia": 113865.2, "silhouette": 0.3951, "davies_bouldin": 0.9551},
        {"k": 6, "inertia": 96211.5,  "silhouette": 0.4053, "davies_bouldin": 0.9079},
        {"k": 7, "inertia": 82136.9,  "silhouette": 0.3775, "davies_bouldin": 0.8711},
        {"k": 8, "inertia": 71774.4,  "silhouette": 0.4164, "davies_bouldin": 0.8325},
    ]

    return {
        "df": df,
        "profiles": profiles,
        "meta": CLUSTER_META,
        "sil": sil, "db": db, "ch": ch,
        "valid_median": valid_median,
        "features": features,
        "pc1_var": round(float(M["clust"]["pca"].explained_variance_ratio_[0] * 100), 2),
        "pc2_var": round(float(M["clust"]["pca"].explained_variance_ratio_[1] * 100), 2),
        "elbow": elbow,
        "total_raw": len(df_raw),
        "K": K,
    }


# ─────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    font_color="#1F1E1D",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1F1E1D", size=11)),
    xaxis=dict(gridcolor="#E8E6DC", color="#6B6A68", zerolinecolor="#E8E6DC", linecolor="#E8E6DC"),
    yaxis=dict(gridcolor="#E8E6DC", color="#6B6A68", zerolinecolor="#E8E6DC", linecolor="#E8E6DC"),
)

def get_layout(**kwargs):
    """Safely combine PLOTLY_LAYOUT with overrides without duplicate kwarg errors."""
    layout = dict(PLOTLY_LAYOUT)
    layout.update(kwargs)
    return layout

CLUSTER_COLORS = {0: "#D97757", 1: "#5A6B7C", 2: "#5B8266", 3: "#877B70"}

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0 14px;border-bottom:1px solid #E8E6DC;margin-bottom:14px;">
  <div style="width:32px;height:32px;background:#FAF9F5;border:1px solid #E8E6DC;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#1F1E1D;font-size:0.85rem;font-weight:700;">ML</div>
  <div>
    <div style="font-weight:600;font-size:0.95rem;color:#1F1E1D;letter-spacing:-0.01em;">ML Platform</div>
    <div style="font-size:0.75rem;color:#6B6A68;">23CSE301 · Amrita VV</div>
  </div>
</div>
""", unsafe_allow_html=True)

    track = st.radio(
        "Module Selection",
        ["Regression (Diamonds)", "Classification (Fraud)", "Clustering (Amazon)"],
        index=0,
        label_visibility="visible",
    )

    st.markdown("---")
    M = load_models()

    if M["reg"]["ready"] and M["clf"]["ready"] and M["clust"]["ready"]:
        st.markdown("""
<div style="background:#FAF9F5;border:1px solid #E8E6DC;border-radius:8px;padding:10px 12px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:6px;color:#2E7D32;font-weight:600;font-size:0.8rem;">
    <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#2E7D32;"></span>
    All Systems Operational
  </div>
  <div style="font-size:0.72rem;color:#6B6A68;margin-top:2px;">10 Regressors · 10 Classifiers · K=4 Clusters</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
**System Status**

Regression: {'Operational' if M['reg']['ready'] else 'Offline'}  
Classification: {'Operational' if M['clf']['ready'] else 'Offline'}  
Clustering: {'Operational' if M['clust']['ready'] else 'Offline'}
""")
        st.warning("Run `python app/train_models.py` to generate model artifacts.")

    st.markdown("<div style='font-size:0.72rem;color:#6B6A68;'>Student ID: CB.SC.U4CSE24125</div>", unsafe_allow_html=True)



# ─────────────────────────────────────────
# ══════════════════════════════════════════
#  TRACK 1: REGRESSION
# ══════════════════════════════════════════
# ─────────────────────────────────────────
if "Regression" in track:

    st.markdown('<div class="kicker">Regression Analysis</div>', unsafe_allow_html=True)
    st.title("Diamond Valuation Engine")
    st.markdown("Predict diamond prices from physical dimensions and quality grades. Evaluated on **53,920 clean diamonds** from the GIA dataset.")

    if not M["reg"]["ready"]:
        st.error(f"Regression models not loaded: `{M['reg'].get('error','unknown')}`\n\nRun `python train_models.py` inside the `gui/` folder first.")
        st.stop()

    reg_data = load_json_metadata("reg_data.json")
    cat_info  = reg_data["categorical_info"]

    tab_pred, tab_compare, tab_charts = st.tabs(["Predict Price", "Model Comparison", "Visualisations"])

    # ── TAB 1: PREDICT ──
    with tab_pred:
        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("##### Diamond Attributes")
            model_choice = st.selectbox("Model", list(M["reg"]["models"].keys()), key="reg_model")

            c1, c2 = st.columns(2)
            with c1:
                carat = st.number_input("Carat Weight", min_value=0.1, max_value=6.0, value=0.90, step=0.01, help="Physical weight of the diamond")
                depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=62.1, step=0.1, help="Total depth / diameter × 100")
                table = st.number_input("Table %", min_value=40.0, max_value=80.0, value=58.0, step=0.1, help="Width of top facet relative to diameter")
                x_val = st.number_input("Length x (mm)", min_value=0.1, max_value=25.0, value=6.11, step=0.01)
            with c2:
                y_val = st.number_input("Width y (mm)", min_value=0.1, max_value=25.0, value=6.09, step=0.01)
                z_val = st.number_input("Height z (mm)", min_value=0.1, max_value=25.0, value=3.79, step=0.01)
                cut     = st.selectbox("Cut Quality", cat_info["cut_options"], index=4)
                color   = st.selectbox("Color Grade (D=best)", cat_info["color_options"], index=3, help="D is colourless (best), J is near-colourless")
            clarity = st.selectbox("Clarity Grade (IF=best)", cat_info["clarity_options"], index=3, help="IF=Internally Flawless (best), I1=Included (worst)")

            volume = round(x_val * y_val * z_val, 3)
            st.caption(f"Computed volume: **{volume} mm³**")

            predict_btn = st.button("Predict Price", type="primary", use_container_width=True, key="reg_predict")

        with col_result:
            st.markdown("##### Valuation Result")

            if predict_btn:
                try:
                    row = pd.DataFrame([{
                        "carat": carat, "depth": depth, "table": table,
                        "x": x_val, "y": y_val, "z": z_val, "volume": volume,
                        "cut": cut, "color": color, "clarity": clarity
                    }])
                    X_proc = M["reg"]["preprocessor"].transform(row)
                    model  = M["reg"]["models"][model_choice]
                    pred   = float(model.predict(X_proc)[0])

                    st.markdown(f"""
<div class="pred-box neutral">
  <div class="pred-label">${pred:,.2f}</div>
  <div class="pred-sub">Estimated value via <strong>{model_choice}</strong></div>
</div>
""", unsafe_allow_html=True)

                    st.markdown("**Input Summary**")
                    st.markdown(f"""
<table class="info-table">
  <tr><td>Carat Weight</td><td>{carat}</td></tr>
  <tr><td>Grade (Cut / Color / Clarity)</td><td>{cut} · {color} · {clarity}</td></tr>
  <tr><td>Dimensions (x · y · z)</td><td>{x_val} × {y_val} × {z_val} mm</td></tr>
  <tr><td>Displacement Volume</td><td>{volume} mm³</td></tr>
  <tr><td>Proportions (Depth / Table)</td><td>{depth}% / {table}%</td></tr>
  <tr><td>Estimator Architecture</td><td>{model_choice}</td></tr>
</table>
""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.info("Fill in diamond specifications and click **Predict Price**.")

                st.markdown("**About the features**")
                st.markdown("""
| Feature | Description |
|---|---|
| Carat | Physical weight (1 ct = 0.2 g) |
| Cut | Shape quality: Fair → Ideal |
| Color | Colour grade: J (faint) → D (colourless) |
| Clarity | Inclusions: I1 (worst) → IF (flawless) |
| Volume | Computed as x·y·z (mm³) |
""")

    # ── TAB 2: MODEL COMPARISON ──
    with tab_compare:
        st.markdown("All models trained on **80% (43,136 diamonds)** and evaluated on **20% (10,784 diamonds)**.")

        models_res = reg_data["models"]
        rows = []
        for name, m in models_res.items():
            rows.append({"Model": name, "R²": m["r2"], "RMSE ($)": m["rmse"], "MAE ($)": m["mae"]})
        df_models = pd.DataFrame(rows).sort_values("R²", ascending=False)

        best_r2 = df_models["R²"].max()
        def highlight_best(row):
            if row["R²"] == best_r2:
                return ["background-color:#FAF0EB; font-weight:600; color:#1F1E1D"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_models.style.apply(highlight_best, axis=1)
                     .format({"R²": "{:.4f}", "RMSE ($)": "{:,.2f}", "MAE ($)": "{:,.2f}"}),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        # Bar chart of R² scores
        fig_r2 = go.Figure(go.Bar(
            x=df_models["R²"], y=df_models["Model"], orientation="h",
            marker_color=["#D97757" if r == best_r2 else "#D5D2C7" for r in df_models["R²"]],
            text=[f"{v:.4f}" for v in df_models["R²"]], textposition="outside",
        ))
        fig_r2.update_layout(title="R² Score Comparison (higher = better)",
                             xaxis_range=[0.85, 1.0], **PLOTLY_LAYOUT)
        st.plotly_chart(fig_r2, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_rmse = go.Figure(go.Bar(
                x=df_models["RMSE ($)"], y=df_models["Model"], orientation="h",
                marker_color=["#D97757" if r == df_models["RMSE ($)"].min() else "#D5D2C7" for r in df_models["RMSE ($)"]],
                text=[f"${v:,.0f}" for v in df_models["RMSE ($)"]], textposition="outside",
            ))
            fig_rmse.update_layout(title="RMSE (lower = better)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_rmse, use_container_width=True)
        with c2:
            fig_mae = go.Figure(go.Bar(
                x=df_models["MAE ($)"], y=df_models["Model"], orientation="h",
                marker_color=["#D97757" if r == df_models["MAE ($)"].min() else "#D5D2C7" for r in df_models["MAE ($)"]],
                text=[f"${v:,.0f}" for v in df_models["MAE ($)"]], textposition="outside",
            ))
            fig_mae.update_layout(title="MAE (lower = better)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_mae, use_container_width=True)

    # ── TAB 3: CHARTS ──
    with tab_charts:
        reg_viz_model = st.selectbox(
            "Select Regression Model for Diagnostic Visualisations",
            list(reg_data["models"].keys()), key="reg_viz_model"
        )
        m_reg_info = reg_data["models"][reg_viz_model]
        avp_all = reg_data.get("actual_vs_pred_by_model", {})
        avp = avp_all.get(reg_viz_model, reg_data["actual_vs_pred"])

        actual_arr = np.array(avp["actual"])
        pred_arr = np.array(avp["predicted"])
        resid_arr = np.array(avp["residuals"])
        abs_err_arr = np.abs(resid_arr)

        mean_err = float(np.mean(resid_arr))
        std_err = float(np.std(resid_arr))

        rk1, rk2, rk3, rk4 = st.columns(4)
        rk1.metric("R² Score", f"{m_reg_info['r2']:.4f}")
        rk2.metric("RMSE", f"${m_reg_info['rmse']:,.2f}")
        rk3.metric("MAE", f"${m_reg_info['mae']:,.2f}")
        rk4.metric("Avg Bias", f"${mean_err:+.1f}", f"σ: ${std_err:,.0f}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=avp["actual"], y=avp["predicted"],
                mode="markers",
                marker=dict(
                    color=abs_err_arr,
                    colorscale=[[0, "#E8E6DC"], [0.5, "#D97757"], [1, "#8C3B1E"]],
                    size=5.5,
                    opacity=0.85,
                    showscale=True,
                    colorbar=dict(title=dict(text="Error ($)", font=dict(color="#6B6A68", size=11)), len=0.75, thickness=12, x=1.02, tickfont=dict(color="#6B6A68"))
                ),
                name="Predictions",
                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<br>Error: $%{marker.color:,.0f}<extra></extra>",
            ))
            max_val = max(max(avp["actual"]), max(avp["predicted"]))
            fig_avp.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines",
                line=dict(color="#C15F3C", dash="dash", width=1.5),
                name="Ideal (y = x)"
            ))
            fig_avp.update_layout(
                title=f"Actual vs. Predicted Price — {reg_viz_model}",
                xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)",
                **get_layout()
            )
            st.plotly_chart(fig_avp, use_container_width=True)

        with c2:
            fig_res = go.Figure(go.Histogram(
                x=avp["residuals"], nbinsx=45,
                marker_color="#D97757", opacity=0.85,
                hovertemplate="Residual: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))
            fig_res.add_vline(x=0, line_color="#C15F3C", line_dash="dash", line_width=1.5)
            fig_res.update_layout(
                title=f"Residual Distribution (Actual − Predicted) — {reg_viz_model}",
                xaxis_title="Residual ($)", yaxis_title="Count",
                **get_layout()
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # Row 2: Diagnostic Heteroscedasticity + Model-Specific Feature/Tier Drivers
        c3, c4 = st.columns(2)
        with c3:
            fig_rvp = go.Figure()
            fig_rvp.add_trace(go.Scatter(
                x=avp["predicted"], y=avp["residuals"],
                mode="markers",
                marker=dict(color="#D97757", opacity=0.7, size=5),
                hovertemplate="Predicted: $%{x:,.0f}<br>Residual: $%{y:,.0f}<extra></extra>",
                name="Residuals"
            ))
            fig_rvp.add_hline(y=0, line_color="#C15F3C", line_dash="dash", line_width=1.5)
            fig_rvp.update_layout(
                title=f"Residuals vs. Predicted Price — {reg_viz_model}",
                xaxis_title="Predicted Price ($)", yaxis_title="Residual ($)",
                **get_layout()
            )
            st.plotly_chart(fig_rvp, use_container_width=True)

        with c4:
            fi_all = reg_data.get("feature_importances_by_model", {})
            if reg_viz_model in fi_all:
                fi_df = pd.DataFrame(fi_all[reg_viz_model][:10]).sort_values("importance")
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                    marker_color="#D97757", opacity=0.9,
                    text=[f"{v:.3f}" if v < 10 else f"{v:,.0f}" for v in fi_df["importance"]],
                    textposition="outside",
                ))
                x_title = "Gini / Split Importance" if ("Forest" in reg_viz_model or "Tree" in reg_viz_model or "Boosting" in reg_viz_model) else "Coefficient Magnitude (|β|)"
                fig_fi.update_layout(
                    title=f"Feature Drivers — {reg_viz_model} (Top 10)",
                    xaxis_title=x_title,
                    **get_layout()
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                tier_names = ["Under $2k", "$2k – $5k", "$5k – $10k", "Over $10k"]
                t1 = float(np.mean(abs_err_arr[actual_arr < 2000])) if any(actual_arr < 2000) else 0.0
                t2 = float(np.mean(abs_err_arr[(actual_arr >= 2000) & (actual_arr < 5000)])) if any((actual_arr >= 2000) & (actual_arr < 5000)) else 0.0
                t3 = float(np.mean(abs_err_arr[(actual_arr >= 5000) & (actual_arr < 10000)])) if any((actual_arr >= 5000) & (actual_arr < 10000)) else 0.0
                t4 = float(np.mean(abs_err_arr[actual_arr >= 10000])) if any(actual_arr >= 10000) else 0.0
                tier_maes = [t1, t2, t3, t4]

                fig_tier = go.Figure(go.Bar(
                    x=tier_names, y=tier_maes,
                    marker_color=["#D5D2C7", "#B8B3A8", "#D97757", "#C15F3C"],
                    text=[f"${v:,.0f}" for v in tier_maes], textposition="outside",
                ))
                fig_tier.update_layout(
                    title=f"Mean Error by Price Tier — {reg_viz_model}",
                    yaxis_title="Average Dollar Error ($)",
                    **get_layout()
                )
                st.plotly_chart(fig_tier, use_container_width=True)


# ─────────────────────────────────────────
# ══════════════════════════════════════════
#  TRACK 2: CLASSIFICATION
# ══════════════════════════════════════════
# ─────────────────────────────────────────
elif "Classification" in track:

    st.markdown('<div class="kicker">Binary Classification</div>', unsafe_allow_html=True)
    st.title("Fraud Transaction Detection")
    st.markdown("Real-time risk scoring across **100,000 financial transactions**. Target: `risk_label` — 0 = Legitimate, 1 = Fraud (~1.5% base prevalence).")

    if not M["clf"]["ready"]:
        st.error(f"Classification models not loaded: `{M['clf'].get('error','unknown')}`\n\nRun `python train_models.py` first.")
        st.stop()

    clf_data = load_json_metadata("clf_data.json")
    ds_info   = clf_data["dataset_info"]
    cat_opts  = clf_data["cat_options"]
    num_feats = clf_data["num_features"]
    cat_feats = clf_data["cat_features"]
    bin_feats = clf_data["bin_features"]

    tab_pred, tab_compare, tab_charts = st.tabs(["Classify Transaction", "Model Comparison", "Visualisations"])

    # ── TAB 1: PREDICT ──
    with tab_pred:
        col_form, col_result = st.columns([1.1, 0.9], gap="large")

        with col_form:
            st.markdown("##### Transaction Details")
            clf_model = st.selectbox("Classifier", list(M["clf"]["models"].keys()), key="clf_model")

            c1, c2 = st.columns(2)
            with c1:
                txn_amt   = st.number_input("Transaction Amount ($)", min_value=0.01, value=250.0, step=1.0)
                avg_amt   = st.number_input("Customer Avg. Txn Amount 30d ($)", min_value=0.0, value=180.0, step=1.0)
                risk_scr  = st.number_input("Customer Risk Score (0–100)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
                acct_age  = st.number_input("Account Age (days)", min_value=0, value=800, step=1)
                chargebacks = st.number_input("Previous Chargebacks", min_value=0, value=0, step=1)
                txn_30d   = st.number_input("Customer Txns Last 30d", min_value=0, value=12, step=1)
            with c2:
                vel_1h    = st.number_input("Transactions Last 1h", min_value=0, value=1, step=1)
                vel_24h   = st.number_input("Transactions Last 24h", min_value=0, value=3, step=1)
                txn_hour  = st.number_input("Transaction Hour (0–23)", min_value=0, max_value=23, value=14, step=1)
                txn_dow   = st.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=2, step=1)
                merchant_cat  = st.selectbox("Merchant Category", cat_opts.get("merchant_category", []))
                device_type   = st.selectbox("Device Type", cat_opts.get("device_type", []))

            c3, c4 = st.columns(2)
            with c3:
                txn_country = st.selectbox("Country", cat_opts.get("transaction_country", []))
                txn_type    = st.selectbox("Transaction Type", cat_opts.get("transaction_type", []))
            with c4:
                geo_region  = st.selectbox("Region", cat_opts.get("geo_location_region", []))

            st.markdown("**Binary Indicators**")
            bc1, bc2, bc3 = st.columns(3)
            is_intl      = bc1.checkbox("International", value=False)
            is_highrisk  = bc2.checkbox("High-Risk Merchant", value=False)
            is_weekend   = bc3.checkbox("Weekend Transaction", value=False)

            amount_ratio = txn_amt / (avg_amt + 1)
            st.caption(f"Computed `amount_ratio` = **{amount_ratio:.3f}** (current ÷ 30-day avg)")

            clf_btn = st.button("Classify Transaction", type="primary", use_container_width=True, key="clf_predict")

        with col_result:
            st.markdown("##### Classification Assessment")

            if clf_btn:
                try:
                    row = pd.DataFrame([{
                        "transaction_hour": txn_hour, "transaction_day_of_week": txn_dow,
                        "account_age_days": acct_age, "previous_chargebacks": chargebacks,
                        "customer_total_transactions_30d": txn_30d, "customer_risk_score": risk_scr,
                        "transaction_amount": txn_amt, "avg_transaction_amount_30d_customer": avg_amt,
                        "transaction_velocity_1h": vel_1h, "transaction_velocity_24h": vel_24h,
                        "amount_ratio": amount_ratio,
                        "merchant_category": merchant_cat, "transaction_country": txn_country,
                        "device_type": device_type, "transaction_type": txn_type,
                        "geo_location_region": geo_region,
                        "is_international": int(is_intl),
                        "is_high_risk_merchant_category": int(is_highrisk),
                        "is_weekend": int(is_weekend),
                    }])[num_feats + cat_feats + bin_feats]

                    X_proc = M["clf"]["preprocessor"].transform(row)
                    model  = M["clf"]["models"][clf_model]
                    pred_class = int(model.predict(X_proc)[0])
                    probs = model.predict_proba(X_proc)[0]
                    fraud_pct = round(probs[1] * 100, 1)
                    legit_pct = round(probs[0] * 100, 1)

                    is_fraud = pred_class == 1
                    box_cls  = "danger" if is_fraud else "success"
                    label    = "High Risk: Fraud Flagged" if is_fraud else "Low Risk: Legitimate Transaction"
                    risk_lvl = "High" if fraud_pct > 70 else ("Medium" if fraud_pct > 30 else "Low")

                    st.markdown(f"""
<div class="pred-box {box_cls}">
  <div class="pred-label">{label}</div>
  <div class="pred-sub">Assessment: <strong>{risk_lvl} Risk</strong> ({fraud_pct}% fraud confidence) · Model: {clf_model}</div>
</div>
""", unsafe_allow_html=True)

                    # Probability bars
                    fig_prob = go.Figure(go.Bar(
                        x=["Legitimate", "Fraud"], y=[legit_pct, fraud_pct],
                        marker_color=["#5B8266", "#D97757"],
                        text=[f"{legit_pct}%", f"{fraud_pct}%"], textposition="outside",
                    ))
                    fig_prob.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 110],
                                           title="Class Probabilities",
                                           **get_layout(margin=dict(l=30, r=20, t=40, b=20)))
                    st.plotly_chart(fig_prob, use_container_width=True)

                    st.markdown(f"""
<table class="info-table">
  <tr><td>Transaction Amount</td><td>${txn_amt:,.2f}</td></tr>
  <tr><td>30-Day Avg Amount</td><td>${avg_amt:,.2f}</td></tr>
  <tr><td>Amount Ratio</td><td>{amount_ratio:.3f}</td></tr>
  <tr><td>Customer Risk Score</td><td>{risk_scr}</td></tr>
  <tr><td>Account Age</td><td>{acct_age} days</td></tr>
  <tr><td>Previous Chargebacks</td><td>{chargebacks}</td></tr>
  <tr><td>Velocity (1h / 24h)</td><td>{vel_1h} / {vel_24h}</td></tr>
</table>
""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Classification error: {e}")
            else:
                st.info("Enter transaction attributes and click **Classify Transaction**.")

                st.markdown("**Class Definitions**")
                st.markdown("""
| Class | Status | Description |
|:---:|---|---|
| 0 | Legitimate | Standard transaction within customer baseline profile |
| 1 | Fraud | Anomalous transaction flagged for security review |
""")
                st.markdown("**Dataset Balance**")
                fig_bal = go.Figure(go.Pie(
                    labels=["Legitimate (98.5%)", "Fraud (1.5%)"],
                    values=[98500, 1500], hole=0.5,
                    marker_colors=["#E8E6DC", "#D97757"],
                ))
                fig_bal.update_layout(showlegend=True,
                                       **get_layout(margin=dict(l=10, r=10, t=10, b=10)))
                st.plotly_chart(fig_bal, use_container_width=True)

    # ── TAB 2: MODEL COMPARISON ──
    with tab_compare:
        st.markdown("All classifiers evaluated on **20,000 held-out transactions** (stratified 80/20 split).")
        models_res = clf_data["models"]
        rows = []
        for name, m in models_res.items():
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Weighted F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
            })
        df_clf_tbl = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        best_auc = df_clf_tbl["ROC-AUC"].max()

        def highlight_clf(row):
            if row["ROC-AUC"] == best_auc:
                return ["background-color:#FAF0EB; font-weight:600; color:#1F1E1D"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_clf_tbl.style.apply(highlight_clf, axis=1)
                      .format({col: "{:.4f}" for col in ["Accuracy","Precision","Recall","Weighted F1","ROC-AUC"]}),
            use_container_width=True, hide_index=True
        )

        # Grouped bar chart
        fig_clf_bar = go.Figure()
        metrics_to_show = ["Accuracy", "Precision", "Recall", "Weighted F1", "ROC-AUC"]
        colors = ["#D97757", "#5B8266", "#D5D2C7", "#8C827A", "#C15F3C"]
        for i, metric in enumerate(metrics_to_show):
            fig_clf_bar.add_trace(go.Bar(
                name=metric, x=df_clf_tbl["Model"], y=df_clf_tbl[metric],
                marker_color=colors[i], opacity=0.85,
            ))
        fig_clf_bar.update_layout(
            barmode="group", title="Model Metrics Comparison",
            yaxis_title="Score", yaxis_range=[0.9, 1.02],
            xaxis_tickangle=-30, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_clf_bar, use_container_width=True)

    # ── TAB 3: CHARTS ──
    with tab_charts:
        sel_c1, sel_c2 = st.columns([2, 1])
        with sel_c1:
            model_for_charts = st.selectbox(
                "Select Classifier for Diagnostic Visualisations",
                list(clf_data["models"].keys()), key="clf_chart_model"
            )
        with sel_c2:
            overlay_all_roc = st.checkbox("Overlay All Models on ROC Curve", value=False, key="clf_overlay_roc")

        m_res = clf_data["models"][model_for_charts]

        # Model Performance KPI strip for selected model
        pk1, pk2, pk3, pk4, pk5 = st.columns(5)
        pk1.metric("Accuracy", f"{m_res['accuracy'] * 100:.2f}%")
        pk2.metric("Precision", f"{m_res['precision'] * 100:.2f}%")
        pk3.metric("Recall", f"{m_res['recall'] * 100:.2f}%")
        pk4.metric("Weighted F1", f"{m_res['f1']:.4f}")
        pk5.metric("ROC-AUC", f"{m_res['roc_auc']:.4f}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            cm = np.array(m_res["confusion_matrix"])
            total_legit = cm[0][0] + cm[0][1]
            total_fraud = cm[1][0] + cm[1][1]
            tn_pct = (cm[0][0] / total_legit * 100) if total_legit else 0
            fp_pct = (cm[0][1] / total_legit * 100) if total_legit else 0
            fn_pct = (cm[1][0] / total_fraud * 100) if total_fraud else 0
            tp_pct = (cm[1][1] / total_fraud * 100) if total_fraud else 0

            text_vals = [
                [f"<b>TN: {cm[0][0]:,}</b><br>({tn_pct:.1f}% of legit)", f"<b>FP: {cm[0][1]:,}</b><br>({fp_pct:.1f}% false alarms)"],
                [f"<b>FN: {cm[1][0]:,}</b><br>({fn_pct:.1f}% missed)", f"<b>TP: {cm[1][1]:,}</b><br>({tp_pct:.1f}% caught)"]
            ]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Predicted: Legit", "Predicted: Fraud"],
                y=["Actual: Legit", "Actual: Fraud"],
                colorscale=[[0, "#FFFFFF"], [0.5, "#FBEBE4"], [1, "#D97757"]],
                text=text_vals, texttemplate="%{text}",
                textfont={"size": 13, "color": "#1F1E1D"},
                showscale=False,
            ))
            fig_cm.update_layout(title=f"Confusion Matrix — {model_for_charts}", **get_layout(height=340))
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            fig_roc = go.Figure()
            if overlay_all_roc:
                for other_name, other_m in clf_data["models"].items():
                    if other_name != model_for_charts:
                        fig_roc.add_trace(go.Scatter(
                            x=other_m["roc_fpr"], y=other_m["roc_tpr"], mode="lines",
                            line=dict(color="#D5D2C7", width=1),
                            name=f"{other_name} ({other_m['roc_auc']:.3f})",
                            opacity=0.6, hoverinfo="name"
                        ))

            fpr = m_res["roc_fpr"]
            tpr = m_res["roc_tpr"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", fill="tozeroy",
                line=dict(color="#D97757", width=2),
                name=f"{model_for_charts} (AUC = {m_res['roc_auc']:.4f})",
                fillcolor="rgba(217, 119, 87, 0.08)"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#6B6A68", dash="dash", width=1),
                name="Random Baseline", showlegend=True
            ))
            fig_roc.update_layout(
                title=f"ROC Curve — {model_for_charts}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **get_layout(height=340)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Row 2: Diagnostic outcome breakdown & Feature Importances/Weights
        c3, c4 = st.columns(2)
        with c3:
            cm = np.array(m_res["confusion_matrix"])
            outcome_names = ["True Legitimate (TN)", "Fraud Caught (TP)", "False Alarm (FP)", "Missed Fraud (FN)"]
            outcome_counts = [int(cm[0][0]), int(cm[1][1]), int(cm[0][1]), int(cm[1][0])]
            outcome_colors = ["#5B8266", "#D97757", "#D5D2C7", "#C15F3C"]

            fig_breakdown = go.Figure(go.Bar(
                x=outcome_names, y=outcome_counts,
                marker_color=outcome_colors,
                text=[f"{v:,}" for v in outcome_counts],
                textposition="outside",
            ))
            fig_breakdown.update_layout(
                title=f"Prediction Breakdown — {model_for_charts}",
                yaxis_title="Transactions",
                xaxis_tickangle=-15,
                **get_layout(height=360)
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)

        with c4:
            fi_models = clf_data.get("feature_importances_by_model", {})
            if model_for_charts in fi_models:
                m_fi = fi_models[model_for_charts]
                top_fi = pd.DataFrame(m_fi[:12]).sort_values("importance")
                fig_fi = go.Figure(go.Bar(
                    x=top_fi["importance"], y=top_fi["feature"], orientation="h",
                    marker_color="#D97757", opacity=0.9,
                    text=[f"{v:.4f}" for v in top_fi["importance"]], textposition="outside",
                ))
                x_label = "Coefficient Magnitude (|β|)" if "Logistic" in model_for_charts else "Gini / Split Importance"
                fig_fi.update_layout(
                    title=f"Feature Drivers — {model_for_charts} (Top 12)",
                    xaxis_title=x_label,
                    **get_layout(height=360)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                all_metrics = ["Accuracy", "Precision", "Recall", "Weighted F1", "ROC-AUC"]
                model_vals = [m_res["accuracy"], m_res["precision"], m_res["recall"], m_res["f1"], m_res["roc_auc"]]
                avg_vals = [
                    float(np.mean([m["accuracy"] for m in clf_data["models"].values()])),
                    float(np.mean([m["precision"] for m in clf_data["models"].values()])),
                    float(np.mean([m["recall"] for m in clf_data["models"].values()])),
                    float(np.mean([m["f1"] for m in clf_data["models"].values()])),
                    float(np.mean([m["roc_auc"] for m in clf_data["models"].values()])),
                ]

                fig_bench = go.Figure()
                fig_bench.add_trace(go.Bar(
                    name=model_for_charts, x=all_metrics, y=model_vals,
                    marker_color="#D97757", opacity=0.9,
                    text=[f"{v:.3f}" for v in model_vals], textposition="outside"
                ))
                fig_bench.add_trace(go.Bar(
                    name="10-Model Average", x=all_metrics, y=avg_vals,
                    marker_color="#D5D2C7", opacity=0.9,
                    text=[f"{v:.3f}" for v in avg_vals], textposition="outside"
                ))
                fig_bench.update_layout(
                    barmode="group",
                    title=f"{model_for_charts} vs. Classifiers Average",
                    yaxis_title="Score",
                    yaxis_range=[0.85, 1.05],
                    **get_layout(height=360)
                )
                st.plotly_chart(fig_bench, use_container_width=True)


# ─────────────────────────────────────────
# ══════════════════════════════════════════
#  TRACK 3: CLUSTERING
# ══════════════════════════════════════════
# ─────────────────────────────────────────
elif "Clustering" in track:

    st.markdown('<div class="kicker">Unsupervised Segmentation</div>', unsafe_allow_html=True)
    st.title("Catalog Product Segmentation")
    st.markdown("K-Means clustering (k=4) on **47,921 Amazon retail products** across pricing, social proof, velocity, and promotional metrics.")

    if not M["clust"]["ready"]:
        st.error(f"Clustering model not loaded: `{M['clust'].get('error','unknown')}`\n\nRun `python train_models.py` first.")
        st.stop()

    C = load_clustering_data()

    tab_classify, tab_analysis, tab_pca, tab_catalog = st.tabs([
        "Classify Product", "Cluster Analysis", "PCA & Diagnostics", "Catalog Explorer"
    ])

    # ── TAB 1: CLASSIFY PRODUCT ──
    with tab_classify:
        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("##### Product Attributes")

            # Quick presets
            preset = st.selectbox("Quick preset", [
                "— Custom —",
                "High-Velocity Bestseller",
                "Discount-Driven Item",
                "Standard Full-Price Item",
                "New / Unrated Cold-Start Listing",
            ], key="clust_preset")

            PRESETS = {
                "High-Velocity Bestseller":       (24.99, 0.0,  4.7, 8200, 3100),
                "Discount-Driven Item":          (39.99, 79.99, 4.2, 1200, 320),
                "Standard Full-Price Item":       (19.99, 0.0,  4.5, 640,  150),
                "New / Unrated Cold-Start Listing": (129.99, 0.0, 0.0, 0,    0),
            }

            def_price, def_list, def_stars, def_rev, def_bought = (
                PRESETS[preset] if preset in PRESETS else (19.99, 29.99, 4.5, 640, 150)
            )

            price      = st.number_input("Selling Price ($)", min_value=0.01, max_value=5000.0, value=def_price, step=0.01)
            list_price = st.number_input("List Price / MSRP ($) — enter 0 if none", min_value=0.0, max_value=5000.0, value=def_list, step=0.01)
            stars      = st.number_input("Customer Rating (0–5, enter 0 = no ratings)", min_value=0.0, max_value=5.0, value=def_stars, step=0.1)
            reviews    = st.number_input("Total Review Count", min_value=0, value=int(def_rev), step=1)
            bought     = st.number_input("Purchases in Last Month", min_value=0, value=int(def_bought), step=1)

            has_rating   = stars > 0
            has_discount = list_price > 0 and list_price > price
            stars_imp    = stars if stars > 0 else C["valid_median"]

            st.caption(f"`has_rating` = **{'Yes' if has_rating else 'No'}** · `has_discount` = **{'Yes' if has_discount else 'No'}** · `stars_imputed` = **{stars_imp:.2f}**")

            clust_btn = st.button("Assign Product to Cluster", type="primary", use_container_width=True, key="clust_predict")

        with col_result:
            st.markdown("##### Cluster Assignment")

            if clust_btn:
                if price <= 0:
                    st.error("Selling price must be > $0.")
                else:
                    try:
                        log_price   = np.log1p(price)
                        log_reviews = np.log1p(reviews)
                        log_bought  = np.log1p(bought)
                        raw_vec = np.array([[log_price, log_reviews, log_bought,
                                             stars_imp, int(has_rating), int(has_discount)]])
                        scaled_vec = M["clust"]["scaler"].transform(raw_vec)
                        pred_c = int(M["clust"]["kmeans"].predict(scaled_vec)[0])

                        distances = []
                        for cid in range(C["K"]):
                            centroid = M["clust"]["kmeans"].cluster_centers_[cid]
                            dist = float(np.linalg.norm(scaled_vec[0] - centroid))
                            distances.append({"cluster": cid, "name": C["meta"][cid]["name"],
                                              "dist": dist, "color": C["meta"][cid]["color"]})
                        max_dist = max(d["dist"] for d in distances)
                        nearest_dist = distances[pred_c]["dist"]
                        confidence = max(10, min(99, int(100 / (1.0 + nearest_dist * 0.4))))

                        meta = C["meta"][pred_c]
                        profile = C["profiles"][pred_c]

                        st.markdown(f"""
<div class="pred-box neutral" style="border-left: 3px solid {meta['color']};">
  <div class="cluster-badge" style="color:{meta['color']}; border-color:{meta['color']};">
    Cluster {pred_c} · {meta['tag']}
  </div>
  <div class="pred-label" style="color:#1F1E1D;">{meta['name']}</div>
  <div class="pred-sub">Centroid Confidence: <strong>{confidence}%</strong> · Nearest Distance: {nearest_dist:.3f}</div>
</div>
""", unsafe_allow_html=True)

                        # Confidence bar
                        fig_conf = go.Figure(go.Bar(
                            x=["Confidence"], y=[confidence],
                            marker_color=meta["color"],
                            text=[f"{confidence}%"], textposition="outside",
                        ))
                        fig_conf.update_layout(yaxis_range=[0, 110], showlegend=False,
                                               title="Assignment Confidence",
                                               **get_layout(margin=dict(l=20,r=20,t=40,b=20), height=180))
                        st.plotly_chart(fig_conf, use_container_width=True)

                        # Distances to all centroids
                        st.markdown("**Euclidean Distances to Centroids**")
                        for d in sorted(distances, key=lambda x: x["dist"]):
                            bar_pct = int((1 - d["dist"] / (max_dist + 0.001)) * 100)
                            nearest_mark = " ← nearest" if d["cluster"] == pred_c else ""
                            st.markdown(f"""
<div class="dist-row">
  <div class="dist-name" title="{d['name']}">C{d['cluster']}: {d['name']}</div>
  <div class="dist-bar-wrap">
    <div class="dist-bar" style="width:{bar_pct}%; background:{d['color']};"></div>
  </div>
  <div class="dist-val">{d['dist']:.3f}{nearest_mark}</div>
</div>
""", unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("**Cluster Profile**")
                        st.markdown(f"""
<table class="info-table">
  <tr><td>Products in Cluster</td><td>{profile['count']:,} ({profile['share']}%)</td></tr>
  <tr><td>Median Price</td><td>${profile['median_price']:.2f}</td></tr>
  <tr><td>Mean Rating</td><td>{profile['mean_stars']:.2f} / 5.00</td></tr>
  <tr><td>Mean Review Count</td><td>{profile['mean_reviews']:,.0f}</td></tr>
  <tr><td>Avg Monthly Purchases</td><td>{profile['mean_bought']:,.0f}</td></tr>
  <tr><td>Discount Rate</td><td>{profile['discount_rate']}%</td></tr>
  <tr><td>Has-Rating Rate</td><td>{profile['rated_rate']}%</td></tr>
</table>
""", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Clustering error: {e}")
            else:
                st.info("Select a preset or enter product attributes, then click **Assign Product to Cluster**.")

                st.markdown("**Segment Definitions**")
                for cid, meta in C["meta"].items():
                    p = C["profiles"][cid]
                    st.markdown(f"""
<div class="clu-card" style="border-left: 3px solid {meta['color']};">
  <div style="font-weight:600; color:{meta['color']}; font-size:0.88rem;">Cluster {cid} — {meta['name']}</div>
  <div style="color:#6B6A68; font-size:0.8rem; margin-top:3px;">{p['count']:,} products ({p['share']}%) · Median ${p['median_price']:.2f} · Discount rate {p['discount_rate']}%</div>
</div>
""", unsafe_allow_html=True)

    # ── TAB 2: CLUSTER ANALYSIS ──
    with tab_analysis:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clean Products", f"{len(C['df']):,}")
        k2.metric("Silhouette Score", f"{C['sil']:.4f}")
        k3.metric("Davies–Bouldin", f"{C['db']:.4f}")
        k4.metric("Calinski–Harabasz", f"{C['ch']:,.0f}")

        # Cluster profile cards
        cols = st.columns(4)
        for cid in range(C["K"]):
            p    = C["profiles"][cid]
            meta = C["meta"][cid]
            with cols[cid]:
                st.markdown(f"""
<div class="clu-card" style="border-top: 3px solid {meta['color']}; text-align:center;">
  <div style="font-size:1.5rem; font-weight:700; color:{meta['color']};">{p['share']}%</div>
  <div style="font-weight:600; color:#1F1E1D; font-size:0.85rem; margin:4px 0;">{meta['name']}</div>
  <div style="color:#6B6A68; font-size:0.75rem;">{p['count']:,} items</div>
  <hr style="border-color:#E8E6DC; margin:10px 0;">
  <div style="font-size:0.78rem; color:#6B6A68; line-height:1.6;">
    Median ${p['median_price']:.0f}<br>
    Rating: {p['mean_stars']:.1f} / 5.0<br>
    {p['mean_reviews']:,.0f} reviews avg<br>
    {p['mean_bought']:,.0f}/mo velocity
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            # Cluster size pie
            fig_pie = go.Figure(go.Pie(
                labels=[f"C{cid}: {C['meta'][cid]['name']}" for cid in range(C["K"])],
                values=[C["profiles"][cid]["count"] for cid in range(C["K"])],
                hole=0.4,
                marker_colors=[C["meta"][cid]["color"] for cid in range(C["K"])],
            ))
            fig_pie.update_layout(title="Cluster Size Distribution", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            # Feature comparison radar-style bar
            feature_metrics = ["mean_price", "mean_stars", "discount_rate", "rated_rate"]
            labels          = ["Avg Price ($)", "Avg Stars", "Discount %", "Rated %"]
            fig_feat = go.Figure()
            for cid in range(C["K"]):
                p = C["profiles"][cid]
                # Normalise price to 0-100 scale for display
                norm_price = min(p["mean_price"] / 5, 100)
                vals = [norm_price, p["mean_stars"] * 20, p["discount_rate"], p["rated_rate"]]
                fig_feat.add_trace(go.Bar(
                    name=f"C{cid}: {C['meta'][cid]['tag']}",
                    x=labels, y=vals,
                    marker_color=C["meta"][cid]["color"], opacity=0.85,
                ))
            fig_feat.update_layout(barmode="group",
                                    title="Normalised Feature Profile by Cluster",
                                    yaxis_title="Score (0–100)",
                                    **PLOTLY_LAYOUT)
            st.plotly_chart(fig_feat, use_container_width=True)

        # Top categories per cluster
        st.markdown("#### Category Distribution by Cluster")
        c_cols = st.columns(4)
        for cid in range(C["K"]):
            with c_cols[cid]:
                meta = C["meta"][cid]
                st.markdown(f"**<span style='color:{meta['color']}'>Cluster {cid}</span>**", unsafe_allow_html=True)
                cats = C["profiles"][cid]["top_categories"]
                for cat_name, cnt in list(cats.items())[:5]:
                    pct = round(cnt / C["profiles"][cid]["count"] * 100, 1)
                    st.progress(int(pct), text=f"{cat_name[:22]}: {pct}%")

    # ── TAB 3: PCA & DIAGNOSTICS ──
    with tab_pca:
        # PCA scatter
        df_full = C["df"]
        sample_dfs = []
        for cid in range(C["K"]):
            sub = df_full[df_full["Cluster"] == cid]
            sample_dfs.append(sub.sample(n=min(200, len(sub)), random_state=42))
        df_sample = pd.concat(sample_dfs)

        fig_pca = go.Figure()
        for cid in range(C["K"]):
            sub = df_sample[df_sample["Cluster"] == cid]
            meta = C["meta"][cid]
            fig_pca.add_trace(go.Scatter(
                x=sub["pca_x"], y=sub["pca_y"],
                mode="markers",
                name=f"C{cid}: {meta['name']}",
                marker=dict(color=meta["color"], size=5, opacity=0.65),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Price: $%{customdata[1]:.2f}<br>"
                    "Stars: %{customdata[2]:.1f}<br>"
                    "Reviews: %{customdata[3]:,}<extra></extra>"
                ),
                customdata=sub[["title", "price", "stars", "reviews"]].values,
            ))
        fig_pca.update_layout(
            title=f"PCA 2D Projection (PC1={C['pc1_var']}% + PC2={C['pc2_var']}% = {round(C['pc1_var']+C['pc2_var'],1)}% variance)",
            xaxis_title=f"PC1 ({C['pc1_var']}% variance)",
            yaxis_title=f"PC2 ({C['pc2_var']}% variance)",
            height=500, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # Elbow curve
        elbow_df = pd.DataFrame(C["elbow"])
        elbow_metric = st.radio("Elbow metric", ["inertia", "silhouette", "davies_bouldin"],
                                 horizontal=True, format_func=lambda x: x.replace("_", " ").title())

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow_df["k"], y=elbow_df[elbow_metric], mode="lines+markers",
            line=dict(color="#D97757", width=2),
            marker=dict(size=8, color=["#C15F3C" if k == 4 else "#D97757" for k in elbow_df["k"]]),
            hovertemplate="k=%{x}<br>" + elbow_metric + "=%{y:.4f}<extra></extra>",
        ))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#C15F3C", line_width=1.5,
                             annotation_text="Selected k=4", annotation_position="top right")
        fig_elbow.update_layout(
            title=f"{elbow_metric.replace('_',' ').title()} vs. k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title=elbow_metric.replace("_", " ").title(),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Metrics cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette Score", C["sil"], help="Range -1 to 1. Higher = better-separated clusters.")
        m2.metric("Davies–Bouldin Index", C["db"], help="Lower = better. Measures cluster compactness vs separation.")
        m3.metric("Calinski–Harabasz", f"{C['ch']:,.0f}", help="Higher = better. Ratio of between-cluster to within-cluster dispersion.")

    # ── TAB 4: CATALOG EXPLORER ──
    with tab_catalog:
        df_all = C["df"]

        fc1, fc2, fc3, fc4 = st.columns([2.5, 1.5, 1.5, 1.5])
        with fc1:
            search = st.text_input("Search product title", placeholder="e.g. headphones, shoes…", label_visibility="collapsed")
        with fc2:
            cat_filter = st.selectbox("Category", ["All"] + sorted(df_all["category_name"].unique().tolist()), label_visibility="collapsed")
        with fc3:
            clust_filter = st.selectbox("Cluster", ["All", "0 · Standard", "1 · High-Velocity", "2 · Discount", "3 · Cold Start"], label_visibility="collapsed")
        with fc4:
            sort_opt = st.selectbox("Sort by", ["Price ↓", "Price ↑", "Most Reviews", "Top Velocity", "Stars ↓"], label_visibility="collapsed")

        filtered = df_all.copy()
        if search:
            filtered = filtered[filtered["title"].str.lower().str.contains(search.lower(), na=False)]
        if cat_filter != "All":
            filtered = filtered[filtered["category_name"] == cat_filter]
        if clust_filter != "All":
            cid_sel = int(clust_filter.split("·")[0].strip())
            filtered = filtered[filtered["Cluster"] == cid_sel]

        sort_map = {
            "Price ↓": ("price", False), "Price ↑": ("price", True),
            "Most Reviews": ("reviews", False), "Top Velocity": ("boughtInLastMonth", False),
            "Stars ↓": ("stars", False),
        }
        col_s, asc_s = sort_map[sort_opt]
        filtered = filtered.sort_values(col_s, ascending=asc_s)

        PAGE_SIZE = 12
        total = len(filtered)
        max_page = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

        if "catalog_page" not in st.session_state:
            st.session_state.catalog_page = 1
        if search or cat_filter != "All" or clust_filter != "All":
            st.session_state.catalog_page = 1

        page = st.session_state.catalog_page
        start = (page - 1) * PAGE_SIZE
        page_df = filtered.iloc[start:start + PAGE_SIZE]

        st.caption(f"**{total:,}** products found · Page {page} of {max_page}")

        # Product grid (3 columns)
        grid_cols = st.columns(3)
        for i, (_, row) in enumerate(page_df.iterrows()):
            cid = int(row["Cluster"])
            color = C["meta"][cid]["color"]
            with grid_cols[i % 3]:
                disc_badge = f'<span style="background:#FAF9F5;color:#C15F3C;border:1px solid #E8E6DC;padding:1px 6px;border-radius:4px;font-size:0.7rem;margin-left:6px;font-weight:600;">SALE</span>' if row["has_discount"] else ""
                st.markdown(f"""
<div style="border:1px solid #E8E6DC;border-left:3px solid {color};border-radius:8px;padding:14px;margin-bottom:12px;background:#FAF9F5;">
  <div style="font-size:0.75rem;color:{color};font-weight:600;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.04em;">
    C{cid} · {C['meta'][cid]['tag']}{disc_badge}
  </div>
  <div style="font-size:0.85rem;font-weight:600;color:#1F1E1D;margin-bottom:6px;line-height:1.35;height:2.6rem;overflow:hidden;">
    {str(row['title'])[:80]}
  </div>
  <div style="font-size:0.75rem;color:#6B6A68;">{row['category_name']}</div>
  <div style="display:flex;justify-content:space-between;margin-top:10px;font-size:0.82rem;">
    <span style="font-weight:700;color:#1F1E1D;">${row['price']:.2f}</span>
    <span style="color:#6B6A68;">{row['stars']:.1f} ★ · {int(row['reviews']):,} reviews</span>
  </div>
  <div style="font-size:0.72rem;color:#6B6A68;margin-top:3px;">{int(row['boughtInLastMonth']):,} bought/month</div>
</div>
""", unsafe_allow_html=True)

        # Pagination
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("← Previous", disabled=(page <= 1)):
                st.session_state.catalog_page -= 1
                st.rerun()
        with p2:
            st.markdown(f"<div style='text-align:center;color:#6B6A68;padding-top:6px;'>Page {page} of {max_page}</div>", unsafe_allow_html=True)
        with p3:
            if st.button("Next →", disabled=(page >= max_page)):
                st.session_state.catalog_page += 1
                st.rerun()
