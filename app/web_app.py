"""
app.py - ML Capstone Full Application Backend
Three tracks: Regression (Diamonds), Classification (Fraud), Clustering (Amazon)
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) in ("app", "gui") else BASE_DIR

MODELS_DIR = os.path.join(ROOT_DIR, 'models') if os.path.exists(os.path.join(ROOT_DIR, 'models')) else os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets') if os.path.exists(os.path.join(ROOT_DIR, 'datasets')) else os.path.join(BASE_DIR, '..', 'datasets')


app = Flask(__name__, template_folder='templates', static_folder='static')

# ============================================================
# GLOBAL STATE - Loaded at startup
# ============================================================
REG = {}     # regression artifacts
CLF = {}     # classification artifacts
CLUST = {}   # clustering artifacts

def load_all_models():
    import joblib

    # ---------- Regression ----------
    try:
        REG['preprocessor'] = joblib.load(os.path.join(MODELS_DIR, 'reg_preprocessor.pkl'))
        REG['models'] = {
            'Random Forest':                  joblib.load(os.path.join(MODELS_DIR, 'reg_best_model.pkl')),
            'Gradient Boosting':              joblib.load(os.path.join(MODELS_DIR, 'reg_gradient_boosting.pkl')),
            'K-Nearest Neighbors':            joblib.load(os.path.join(MODELS_DIR, 'reg_k_nearest_neighbors.pkl')),
            'Decision Tree':                  joblib.load(os.path.join(MODELS_DIR, 'reg_decision_tree.pkl')),
            'Polynomial Regression':          joblib.load(os.path.join(MODELS_DIR, 'reg_polynomial_regression.pkl')),
            'Ridge Regression':               joblib.load(os.path.join(MODELS_DIR, 'reg_ridge_regression.pkl')),
            'Linear Regression':              joblib.load(os.path.join(MODELS_DIR, 'reg_linear_regression.pkl')),
            'Lasso Regression':               joblib.load(os.path.join(MODELS_DIR, 'reg_lasso_regression.pkl')),
            'ElasticNet':                     joblib.load(os.path.join(MODELS_DIR, 'reg_elasticnet.pkl')),
            'Support Vector Regressor (SVR)': joblib.load(os.path.join(MODELS_DIR, 'reg_support_vector_regressor_svr.pkl')),
        }
        with open(os.path.join(MODELS_DIR, 'reg_data.json')) as f:
            REG['data'] = json.load(f)
        REG['ready'] = True
        print(f"  [OK] Regression models loaded ({len(REG['models'])} models).")
    except Exception as e:
        REG['ready'] = False
        REG['error'] = str(e)
        print(f"  [WARN] Regression not ready: {e}")

    # ---------- Classification ----------
    try:
        CLF['preprocessor'] = joblib.load(os.path.join(MODELS_DIR, 'clf_preprocessor.pkl'))
        CLF['models'] = {
            'Random Forest':          joblib.load(os.path.join(MODELS_DIR, 'clf_best_model.pkl')),
            'Gradient Boosting':      joblib.load(os.path.join(MODELS_DIR, 'clf_gradient_boosting.pkl')),
            'Logistic Regression':    joblib.load(os.path.join(MODELS_DIR, 'clf_logistic_regression.pkl')),
            'AdaBoost':               joblib.load(os.path.join(MODELS_DIR, 'clf_adaboost.pkl')),
            'Decision Tree':          joblib.load(os.path.join(MODELS_DIR, 'clf_decision_tree.pkl')),
            'K-Nearest Neighbors':    joblib.load(os.path.join(MODELS_DIR, 'clf_k-nearest_neighbors.pkl')),
            'Gaussian Naive Bayes':   joblib.load(os.path.join(MODELS_DIR, 'clf_gaussian_naive_bayes.pkl')),
            'Support Vector Machine': joblib.load(os.path.join(MODELS_DIR, 'clf_support_vector_machine.pkl')),
            'Bagging':                joblib.load(os.path.join(MODELS_DIR, 'clf_bagging.pkl')),
            'MLP Neural Network':     joblib.load(os.path.join(MODELS_DIR, 'clf_mlp_neural_network.pkl')),
        }
        with open(os.path.join(MODELS_DIR, 'clf_data.json')) as f:
            CLF['data'] = json.load(f)
        CLF['ready'] = True
        print(f"  [OK] Classification models loaded ({len(CLF['models'])} models).")
    except Exception as e:
        CLF['ready'] = False
        CLF['error'] = str(e)
        print(f"  [WARN] Classification not ready: {e}")

    # ---------- Clustering ----------
    try:
        CLUST['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'clust_scaler.pkl'))
        CLUST['kmeans'] = joblib.load(os.path.join(MODELS_DIR, 'clust_kmeans.pkl'))
        CLUST['pca']    = joblib.load(os.path.join(MODELS_DIR, 'clust_pca.pkl'))
        CLUST['ready'] = True
        print("  [OK] Clustering model loaded.")
        _load_clustering_data()
    except Exception as e:
        CLUST['ready'] = False
        CLUST['error'] = str(e)
        print(f"  [WARN] Clustering not ready: {e}")


def _load_clustering_data():
    """Pre-compute all clustering analytics at startup."""
    import numpy as np

    SEED = 42
    K_FINAL = 4

    csv_path = os.path.join(DATASETS_DIR, 'amzon_sample_pod_cat.csv')
    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw[df_raw['price'] > 0].copy().reset_index(drop=True)

    df_clean['has_rating'] = (df_clean['stars'] > 0).astype(int)
    valid_median = float(df_clean[df_clean['stars'] > 0]['stars'].median())
    df_clean['stars_imputed'] = df_clean['stars'].replace(0, valid_median)
    df_clean['has_discount'] = ((df_clean['listPrice'] > 0) & (df_clean['listPrice'] > df_clean['price'])).astype(int)
    df_clean['log_price'] = np.log1p(df_clean['price'])
    df_clean['log_reviews'] = np.log1p(df_clean['reviews'])
    df_clean['log_bought'] = np.log1p(df_clean['boughtInLastMonth'])

    features = ['log_price', 'log_reviews', 'log_bought', 'stars_imputed', 'has_rating', 'has_discount']
    X_scaled = CLUST['scaler'].transform(df_clean[features].values)
    df_clean['Cluster'] = CLUST['kmeans'].predict(X_scaled)
    X_pca = CLUST['pca'].transform(X_scaled)
    df_clean['pca_x'] = X_pca[:, 0]
    df_clean['pca_y'] = X_pca[:, 1]

    CLUST['valid_median'] = valid_median
    CLUST['df'] = df_clean
    CLUST['features'] = features
    CLUST['K'] = K_FINAL
    CLUST['pc1_var'] = float(CLUST['pca'].explained_variance_ratio_[0] * 100)
    CLUST['pc2_var'] = float(CLUST['pca'].explained_variance_ratio_[1] * 100)

    # Cluster metadata
    CLUST['metadata'] = {
        0: {'name': 'Standard Full-Price Catalog', 'tag': 'Core Evergreen', 'color': '#3b82f6'},
        1: {'name': 'High-Velocity & Reviewed', 'tag': 'Social Proof Star', 'color': '#ef4444'},
        2: {'name': 'Promotional & Discounted', 'tag': 'Discount Driven', 'color': '#10b981'},
        3: {'name': 'Unrated / New / Dormant', 'tag': 'Cold Start / Luxury', 'color': '#8b5cf6'},
    }

    # Pre-compute cluster profiles
    profiles = {}
    for c_id in range(K_FINAL):
        c_df = df_clean[df_clean['Cluster'] == c_id]
        cat_shares = (c_df['category_name'].value_counts().head(6) / len(c_df) * 100).round(1).to_dict()
        profiles[c_id] = {
            'id': c_id,
            'name': CLUST['metadata'][c_id]['name'],
            'tag': CLUST['metadata'][c_id]['tag'],
            'color': CLUST['metadata'][c_id]['color'],
            'count': int(len(c_df)),
            'share': round(float(len(c_df) / len(df_clean) * 100), 2),
            'mean_price': round(float(c_df['price'].mean()), 2),
            'median_price': round(float(c_df['price'].median()), 2),
            'mean_stars': round(float(c_df['stars_imputed'].mean()), 2),
            'mean_reviews': round(float(c_df['reviews'].mean()), 0),
            'median_reviews': round(float(c_df['reviews'].median()), 0),
            'mean_bought': round(float(c_df['boughtInLastMonth'].mean()), 0),
            'discount_rate': round(float(c_df['has_discount'].mean() * 100), 1),
            'rated_rate': round(float(c_df['has_rating'].mean() * 100), 1),
            'top_categories': cat_shares,
        }
    CLUST['profiles'] = profiles

    # Metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    CLUST['sil'] = round(float(silhouette_score(X_scaled, df_clean['Cluster'], sample_size=10000, random_state=SEED)), 4)
    CLUST['db']  = round(float(davies_bouldin_score(X_scaled, df_clean['Cluster'])), 4)
    CLUST['ch']  = round(float(calinski_harabasz_score(X_scaled, df_clean['Cluster'])), 1)

    print(f"    Clustering data loaded: {len(df_clean):,} products, Sil={CLUST['sil']:.4f}")


print("\nInitializing ML Capstone Application...")
load_all_models()
print("Ready.\n")

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


# ===================== REGRESSION API =====================

@app.route('/api/regression/info')
def reg_info():
    if not REG.get('ready'):
        return jsonify({'status': 'error', 'message': REG.get('error', 'Models not loaded')}), 503
    d = REG['data']
    return jsonify({
        'status': 'success',
        'dataset': d['dataset_info'],
        'models': d['models'],
        'best_model': d['best_model'],
        'feature_names': d['feature_names'],
        'feature_importance': d['feature_importance'],
        'categorical_info': d['categorical_info'],
        'actual_vs_pred': d['actual_vs_pred'],
        'available_models': list(REG['models'].keys()),
    })


@app.route('/api/regression/predict', methods=['POST'])
def reg_predict():
    if not REG.get('ready'):
        return jsonify({'status': 'error', 'message': 'Regression models not loaded. Run train_models.py first.'}), 503
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'Random Forest')
        if model_name not in REG['models']:
            return jsonify({'status': 'error', 'message': f'Model "{model_name}" not available.'}), 400

        # Parse and validate inputs
        carat = float(data.get('carat', 0))
        depth = float(data.get('depth', 0))
        table = float(data.get('table', 0))
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        z = float(data.get('z', 0))
        cut = str(data.get('cut', ''))
        color = str(data.get('color', ''))
        clarity = str(data.get('clarity', ''))

        errors = []
        if carat <= 0 or carat > 6: errors.append("Carat must be between 0.1 and 6.")
        if depth <= 0 or depth > 100: errors.append("Depth % must be between 1 and 100.")
        if table <= 0 or table > 100: errors.append("Table % must be between 1 and 100.")
        if x <= 0 or x > 25: errors.append("Length x must be between 0.1 and 25 mm.")
        if y <= 0 or y > 25: errors.append("Width y must be between 0.1 and 25 mm.")
        if z <= 0 or z > 25: errors.append("Height z must be between 0.1 and 25 mm.")
        cat_info = REG['data']['categorical_info']
        if cut not in cat_info['cut_options']: errors.append(f"Invalid cut: {cut}.")
        if color not in cat_info['color_options']: errors.append(f"Invalid color: {color}.")
        if clarity not in cat_info['clarity_options']: errors.append(f"Invalid clarity: {clarity}.")
        if errors:
            return jsonify({'status': 'error', 'message': ' '.join(errors)}), 400

        volume = x * y * z
        row = pd.DataFrame([{
            'carat': carat, 'depth': depth, 'table': table,
            'x': x, 'y': y, 'z': z, 'volume': volume,
            'cut': cut, 'color': color, 'clarity': clarity
        }])
        X_proc = REG['preprocessor'].transform(row)
        model = REG['models'][model_name]
        pred = float(model.predict(X_proc)[0])

        return jsonify({
            'status': 'success',
            'model_used': model_name,
            'predicted_price': round(pred, 2),
            'formatted_price': f"${pred:,.2f}",
            'inputs': {
                'carat': carat, 'cut': cut, 'color': color, 'clarity': clarity,
                'depth': depth, 'table': table, 'x': x, 'y': y, 'z': z, 'volume': round(volume, 2)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===================== CLASSIFICATION API =====================

@app.route('/api/classification/info')
def clf_info():
    if not CLF.get('ready'):
        return jsonify({'status': 'error', 'message': CLF.get('error', 'Models not loaded')}), 503
    d = CLF['data']
    return jsonify({
        'status': 'success',
        'dataset': d['dataset_info'],
        'models': d['models'],
        'best_model': d['best_model'],
        'feature_importance': d['feature_importance'],
        'cat_options': d['cat_options'],
        'num_features': d['num_features'],
        'cat_features': d['cat_features'],
        'bin_features': d['bin_features'],
        'available_models': list(CLF['models'].keys()),
    })


@app.route('/api/classification/predict', methods=['POST'])
def clf_predict():
    if not CLF.get('ready'):
        return jsonify({'status': 'error', 'message': 'Classification models not loaded. Run train_models.py first.'}), 503
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'Random Forest')
        if model_name not in CLF['models']:
            return jsonify({'status': 'error', 'message': f'Model "{model_name}" not available.'}), 400

        d_info = CLF['data']
        num_features = d_info['num_features']
        cat_features = d_info['cat_features']
        bin_features = d_info['bin_features']

        # Build input row
        row = {}
        errors = []
        for feat in num_features:
            if feat == 'amount_ratio':
                continue
            val = data.get(feat)
            if val is None:
                errors.append(f"Missing: {feat}")
            else:
                try:
                    row[feat] = float(val)
                except:
                    errors.append(f"Invalid value for {feat}")

        # Compute amount_ratio (derived feature)
        if 'transaction_amount' in row and 'avg_transaction_amount_30d_customer' in row:
            row['amount_ratio'] = row['transaction_amount'] / (row['avg_transaction_amount_30d_customer'] + 1.0)
        elif 'amount_ratio' in data:
            row['amount_ratio'] = float(data['amount_ratio'])
        else:
            row['amount_ratio'] = 1.0

        cat_opts = d_info['cat_options']
        for feat in cat_features:
            val = data.get(feat)
            if val is None:
                errors.append(f"Missing: {feat}")
            elif val not in cat_opts.get(feat, []):
                errors.append(f"Invalid option for {feat}: {val}")
            else:
                row[feat] = val

        # Binary features (auto-derive or default)
        for feat in bin_features:
            val = data.get(feat)
            if val is not None:
                row[feat] = int(val)
            elif feat == 'is_weekend':
                row[feat] = 1 if row.get('transaction_day_of_week', 0) in [5, 6] else 0
            elif feat == 'is_high_risk_merchant_category':
                row[feat] = 1 if row.get('merchant_category', '') in ['gambling', 'luxury_goods'] else 0
            elif feat == 'is_international':
                row[feat] = 1 if row.get('transaction_country', 'US') != 'US' else 0
            else:
                row[feat] = 0

        if errors:
            return jsonify({'status': 'error', 'message': ', '.join(errors[:3])}), 400

        predictor_cols = num_features + cat_features + bin_features
        X_row = pd.DataFrame([row])[predictor_cols]
        X_proc = CLF['preprocessor'].transform(X_row)

        model = CLF['models'][model_name]
        pred_class = int(model.predict(X_proc)[0])
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_proc)[0]
            fraud_prob = float(probs[1])
            legit_prob = float(probs[0])
        elif hasattr(model, 'decision_function'):
            df_val = float(model.decision_function(X_proc)[0])
            fraud_prob = float(1.0 / (1.0 + np.exp(-df_val)))
            legit_prob = 1.0 - fraud_prob
        else:
            fraud_prob = float(pred_class)
            legit_prob = 1.0 - fraud_prob

        return jsonify({
            'status': 'success',
            'model_used': model_name,
            'prediction': {
                'class': pred_class,
                'label': 'FRAUD' if pred_class == 1 else 'LEGITIMATE',
                'fraud_probability': round(fraud_prob * 100, 1),
                'legitimate_probability': round(legit_prob * 100, 1),
                'risk_level': 'High' if fraud_prob > 0.7 else ('Medium' if fraud_prob > 0.3 else 'Low'),
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===================== CLUSTERING API =====================

@app.route('/api/clustering/overview')
def clust_overview():
    if not CLUST.get('ready'):
        return jsonify({'status': 'error', 'message': CLUST.get('error', 'Clustering not ready')}), 503

    df = CLUST['df']
    return jsonify({
        'status': 'success',
        'dataset': {
            'total_raw': int(len(df) + int((~(df['price'] > 0)).sum())),
            'clean_records': int(len(df)),
            'categories': sorted(df['category_name'].unique().tolist()),
        },
        'model': {
            'algorithm': 'K-Means',
            'k': CLUST['K'],
            'silhouette': CLUST['sil'],
            'davies_bouldin': CLUST['db'],
            'calinski_harabasz': CLUST['ch'],
            'pca_pc1_var': round(CLUST['pc1_var'], 2),
            'pca_pc2_var': round(CLUST['pc2_var'], 2),
        },
        'clusters': list(CLUST['profiles'].values()),
        'elbow_data': [
            {'k': 2, 'inertia': 230021.3, 'silhouette': 0.3053, 'davies_bouldin': 1.6180},
            {'k': 3, 'inertia': 182768.0, 'silhouette': 0.3612, 'davies_bouldin': 1.2608},
            {'k': 4, 'inertia': 144718.8, 'silhouette': 0.3508, 'davies_bouldin': 1.0759},
            {'k': 5, 'inertia': 113865.2, 'silhouette': 0.3951, 'davies_bouldin': 0.9551},
            {'k': 6, 'inertia': 96211.5,  'silhouette': 0.4053, 'davies_bouldin': 0.9079},
            {'k': 7, 'inertia': 82136.9,  'silhouette': 0.3775, 'davies_bouldin': 0.8711},
            {'k': 8, 'inertia': 71774.4,  'silhouette': 0.4164, 'davies_bouldin': 0.8325},
        ],
        'selected_k': CLUST['K'],
    })


@app.route('/api/clustering/pca_points')
def clust_pca():
    if not CLUST.get('ready'):
        return jsonify({'status': 'error', 'message': 'Clustering not ready'}), 503

    df = CLUST['df']
    K = CLUST['K']
    sampled = []
    for c_id in range(K):
        sub = df[df['Cluster'] == c_id]
        n = min(150, len(sub))
        sampled.append(sub.sample(n=n, random_state=42))
    sample_df = pd.concat(sampled).reset_index(drop=True)

    points = []
    for _, row in sample_df.iterrows():
        points.append({
            'x': round(float(row['pca_x']), 3),
            'y': round(float(row['pca_y']), 3),
            'cluster': int(row['Cluster']),
            'title': str(row['title'])[:55] + '…',
            'price': round(float(row['price']), 2),
            'category': str(row['category_name']),
            'stars': round(float(row['stars']), 1),
            'reviews': int(row['reviews']),
        })

    return jsonify({
        'status': 'success',
        'points': points,
        'pc1_var': round(CLUST['pc1_var'], 2),
        'pc2_var': round(CLUST['pc2_var'], 2),
    })


@app.route('/api/clustering/predict', methods=['POST'])
def clust_predict():
    if not CLUST.get('ready'):
        return jsonify({'status': 'error', 'message': 'Clustering model not loaded. Run train_models.py first.'}), 503
    try:
        data = request.get_json(force=True)
        price = float(data.get('price', 0))
        list_price = float(data.get('listPrice', 0))
        stars = float(data.get('stars', 0))
        reviews = int(data.get('reviews', 0))
        bought = int(data.get('boughtInLastMonth', 0))

        if price <= 0:
            return jsonify({'status': 'error', 'message': 'Selling price must be > 0.'}), 400
        if stars < 0 or stars > 5:
            return jsonify({'status': 'error', 'message': 'Stars must be between 0 and 5.'}), 400

        has_rating = 1 if stars > 0 else 0
        stars_imp = stars if stars > 0 else CLUST['valid_median']
        has_discount = 1 if (list_price > 0 and list_price > price) else 0
        log_price = np.log1p(price)
        log_reviews = np.log1p(reviews)
        log_bought = np.log1p(bought)

        raw_vec = np.array([[log_price, log_reviews, log_bought, stars_imp, has_rating, has_discount]])
        scaled_vec = CLUST['scaler'].transform(raw_vec)
        pred_cluster = int(CLUST['kmeans'].predict(scaled_vec)[0])

        distances = []
        for c_id in range(CLUST['K']):
            centroid = CLUST['kmeans'].cluster_centers_[c_id]
            dist = float(np.linalg.norm(scaled_vec[0] - centroid))
            distances.append({'cluster': c_id, 'name': CLUST['metadata'][c_id]['name'],
                               'distance': round(dist, 3), 'is_nearest': (c_id == pred_cluster)})

        pca_proj = CLUST['pca'].transform(scaled_vec)[0]
        nearest_dist = [d for d in distances if d['is_nearest']][0]['distance']
        confidence = max(10, min(99, int(100 / (1.0 + nearest_dist * 0.4))))

        meta = CLUST['metadata'][pred_cluster]
        profile = CLUST['profiles'][pred_cluster]

        return jsonify({
            'status': 'success',
            'prediction': {
                'cluster': pred_cluster,
                'name': meta['name'],
                'tag': meta['tag'],
                'color': meta['color'],
                'confidence': confidence,
                'profile': profile,
                'centroid_distances': distances,
                'pca_coords': {'x': round(float(pca_proj[0]), 3), 'y': round(float(pca_proj[1]), 3)},
                'input_summary': {
                    'price': price, 'listPrice': list_price, 'stars': stars,
                    'reviews': reviews, 'boughtInLastMonth': bought,
                    'has_rating': bool(has_rating), 'has_discount': bool(has_discount)
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/clustering/catalog')
def clust_catalog():
    if not CLUST.get('ready'):
        return jsonify({'status': 'error', 'message': 'Clustering not ready'}), 503

    df = CLUST['df']
    query = request.args.get('search', '').strip().lower()
    category = request.args.get('category', '').strip()
    cluster_filter = request.args.get('cluster', '').strip()
    sort_by = request.args.get('sort', 'price_desc')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 12))

    filtered = df
    if query:
        filtered = filtered[filtered['title'].str.lower().str.contains(query, na=False)]
    if category and category != 'All':
        filtered = filtered[filtered['category_name'] == category]
    if cluster_filter and cluster_filter != 'All':
        filtered = filtered[filtered['Cluster'] == int(cluster_filter)]

    sort_map = {
        'price_desc': ('price', False), 'price_asc': ('price', True),
        'reviews_desc': ('reviews', False), 'bought_desc': ('boughtInLastMonth', False),
        'stars_desc': ('stars', False),
    }
    if sort_by in sort_map:
        col, asc = sort_map[sort_by]
        filtered = filtered.sort_values(by=col, ascending=asc)

    total = len(filtered)
    start = (page - 1) * page_size
    page_records = filtered.iloc[start:start + page_size]

    items = []
    for _, row in page_records.iterrows():
        c_id = int(row['Cluster'])
        items.append({
            'asin': str(row['asin']),
            'title': str(row['title']),
            'category': str(row['category_name']),
            'price': round(float(row['price']), 2),
            'listPrice': round(float(row['listPrice']), 2),
            'stars': round(float(row['stars']), 1),
            'reviews': int(row['reviews']),
            'boughtInLastMonth': int(row['boughtInLastMonth']),
            'cluster': c_id,
            'cluster_name': CLUST['metadata'][c_id]['name'],
            'cluster_color': CLUST['metadata'][c_id]['color'],
            'has_discount': bool(row['has_discount']),
        })

    return jsonify({
        'status': 'success',
        'total': total,
        'page': page,
        'total_pages': max(1, (total + page_size - 1) // page_size),
        'items': items,
    })


# ===================== HEALTH CHECK =====================

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'regression_ready': REG.get('ready', False),
        'classification_ready': CLF.get('ready', False),
        'clustering_ready': CLUST.get('ready', False),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting ML Capstone Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
