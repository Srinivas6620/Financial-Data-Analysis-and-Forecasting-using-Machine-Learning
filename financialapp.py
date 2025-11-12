"""
streamlit_fraud_full_app_with_animations.py

Bright, colorful Lottie animations for each numbered sidebar page.
Complete Real-Time Credit Card Fraud Detection Dashboard (demo).

Notes:
- Per user request, the old "Threshold Adjustment" page was removed.
- A single "Train All Models" button is provided in the sidebar to train all models at once.
- Numbered quick-navigation buttons are present in the sidebar to jump to pages directly.

Run:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib shap streamlit-lottie
    streamlit run streamlit_fraud_full_app_with_animations.py
"""
import os
import io
import time
import joblib
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Optional SHAP import (guarded)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Optional Lottie support
try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

st.set_page_config(layout='wide', page_title='Real-Time Fraud Detection — Bright')

# ------------------ Configuration ------------------
DEFAULT_DATASET_PATH = Path("C:/Users/teste/OneDrive/Desktop/financial/creditcard.csv.zip")

# Bright & colorful lottie assets (each page has a unique animation)
LOTTIE = {
    # Data
    'upload': 'https://assets8.lottiefiles.com/packages/lf20_3rwasyjy.json',
    'dataset_preview': 'https://assets6.lottiefiles.com/packages/lf20_1b4d2v8u.json',

    # EDA
    'bar_chart': 'https://assets4.lottiefiles.com/packages/lf20_5wW2q4.json',
    'histogram': 'https://assets9.lottiefiles.com/packages/lf20_Z7p6cE.json',
    'scatter': 'https://assets2.lottiefiles.com/packages/lf20_rwqv6w3w.json',
    'heatmap': 'https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json',

    # Outlier / alerts
    'alert': 'https://assets1.lottiefiles.com/packages/lf20_Au7V0f.json',
    'notify': 'https://assets10.lottiefiles.com/packages/lf20_Uw2S0t.json',

    # Model / training
    'training': 'https://assets7.lottiefiles.com/packages/lf20_8kz9q2qv.json',
    'matrix': 'https://assets3.lottiefiles.com/packages/lf20_x62chJ.json',
    'roc': 'https://assets10.lottiefiles.com/packages/lf20_6w4ohwii.json',
    'feature': 'https://assets2.lottiefiles.com/packages/lf20_8gqq5l3y.json',
    'shap': 'https://assets2.lottiefiles.com/packages/lf20_sSF6EG.json',
    'slider': 'https://assets3.lottiefiles.com/packages/lf20_7yq0cr7p.json',
    'compare': 'https://assets6.lottiefiles.com/packages/lf20_6kU9hK.json',

    # Simulation & input
    'live': 'https://assets5.lottiefiles.com/packages/lf20_2g2lxj6v.json',
    'input': 'https://assets1.lottiefiles.com/packages/lf20_5ngs2ksb.json',
    'batch': 'https://assets3.lottiefiles.com/packages/lf20_7yq3y9y6.json',
    'download': 'https://assets7.lottiefiles.com/packages/lf20_usmfx6bp.json',
    'save': 'https://assets2.lottiefiles.com/packages/lf20_vfapq0zr.json',

    # Misc bright ones
    'sparkle': 'https://assets7.lottiefiles.com/packages/lf20_g8x7y9nq.json',
    'colorburst': 'https://assets8.lottiefiles.com/packages/lf20_x62chJ.json'
}

# ------------------ Helpers ------------------

def load_lottie_url(url_or_key):
    """Return lottie JSON from known key or full URL (returns None on failure)."""
    if url_or_key is None:
        return None
    url = LOTTIE.get(url_or_key, url_or_key)
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def safe_st_lottie(key_or_url, height=180):
    """Render Lottie animation if available, otherwise fallback text."""
    lottie_json = load_lottie_url(key_or_url)
    if lottie_json is not None and st_lottie is not None:
        try:
            st_lottie(lottie_json, height=height)
            return
        except Exception:
            pass
    # fallback text badge (bright emoji)
    st.markdown(f"✨ _(animation: {key_or_url})_ ✨")


@st.cache_data
def try_load_from_path(path: str):
    """Attempt to read a CSV or zipped CSV from a disk path. Returns df or None."""
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        if str(p).lower().endswith(".zip"):
            df = pd.read_csv(p, compression='zip')
        else:
            df = pd.read_csv(p)
        return df
    except Exception:
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return None


def load_from_uploaded(uploaded_file):
    """Load a pandas DataFrame from a Streamlit uploaded file (csv or zip)."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        name = getattr(uploaded_file, "name", "")
        if name.lower().endswith(".zip"):
            return pd.read_csv(uploaded_file, compression='zip')
        else:
            return pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            return pd.read_csv(io.BytesIO(raw))
        except Exception:
            return None


def generate_synthetic(n_samples=5000, n_features=30, imbalance=0.005, random_state=42):
    """Generate a simple synthetic fraud-like dataset if no real data is available."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.1),
        n_clusters_per_class=1,
        weights=[1 - imbalance, imbalance],
        flip_y=0,
        random_state=random_state
    )
    if n_features >= 30:
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    else:
        cols = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X[:, :len(cols)], columns=cols)
    df["Class"] = y
    return df


def safe_predict_proba(model, X):
    """Return probabilities for the positive class from a model if possible, else fallback to predict numeric."""
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            try:
                from scipy.special import expit
                return expit(scores)
            except Exception:
                return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            preds = model.predict(X)
            return np.array(preds).astype(float)
    except Exception:
        preds = model.predict(X)
        return np.array(preds).astype(float)


def ensure_session_columns(df):
    """Store canonical feature ordering in session_state for future batch scoring."""
    cols = [c for c in df.columns if c != "Class"]
    st.session_state["feature_columns"] = cols
    return cols


def train_models(X_train, y_train, rf_n=100):
    """Train 3 models and return dict. Not cached because model objects are not always serializable."""
    models = {}
    rf = RandomForestClassifier(n_estimators=rf_n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    gb = GradientBoostingClassifier(n_estimators=100)
    gb.fit(X_train, y_train)
    models["GradientBoosting"] = gb

    return models

# ------------------ App Layout ------------------
st.title("✨ Real-Time Credit Card Fraud Detection — Bright Dashboard ✨")

# Sidebar with grouped and numbered buttons
st.sidebar.markdown("## 🔢 Navigation (click numbers to jump)")
st.sidebar.markdown("### Quick-nav (numbers)")

# Pages list: note "Threshold Adjustment" removed per user request.
pages_data = [
    "1. Dataset Upload & Preview",
    "2. Dataset Summary & Preview"
]
pages_eda = [
    "3. Univariate Analysis",
    "4. Bivariate Analysis",
    "5. Multivariate Analysis",
    "6. Outlier Detection"
]
pages_model = [
    "7. Model Training",
    "8. Confusion Matrix",
    "9. ROC & PR Curves",
    "10. Feature Importance",
    "11. SHAP Explainability",
    # "12. Threshold Adjustment",  # REMOVED per user request
    "12. Model Comparison"  # kept numbering label consistent but now this is the next page
]
pages_sim = [
    "13. Real-Time Simulation",
    "14. Custom Transaction Input",
    "15. Batch Prediction",
    "16. Download Predictions",
    "17. Alerts Simulation",
    "18. Model Save/Load"
]

# Combine pages (final count: 18 pages after removal)
pages = pages_data + pages_eda + pages_model + pages_sim

# Initialize session state for selected page
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = pages[0]
if "page_index" not in st.session_state:
    st.session_state["page_index"] = 0

# Sidebar quick-number buttons (grid)
num_cols = 6
for i in range(0, len(pages), num_cols):
    cols = st.sidebar.columns(min(num_cols, len(pages) - i))
    for j, col in enumerate(cols):
        idx = i + j
        label = str(idx + 1)
        if col.button(label, key=f"num_nav_{idx}"):
            st.session_state["page_index"] = idx
            st.session_state["selected_page"] = pages[idx]
            st.rerun()


st.sidebar.markdown("---")
st.sidebar.markdown("## 🧰 Tools")
# Single one-click train button in sidebar (user asked 'one model training button')
with st.sidebar.expander("Train models (one-click)"):
    st.write("Train all models (RandomForest, LogisticRegression, GradientBoosting) using current dataset.")
    rf_n_sidebar = st.number_input("RandomForest n_estimators", min_value=10, max_value=1000, value=100, step=10, key="rf_n_sidebar")
    test_frac_sidebar = st.slider("Test set fraction", 0.1, 0.5, 0.3, key="test_frac_sidebar")
    if st.button("Train All Models (one-click)"):
        if "df_loaded" not in st.session_state:
            st.warning("Load dataset first from pages 1-2.")
        else:
            df_train = st.session_state["df_loaded"].copy()
            if "Class" not in df_train.columns:
                df_train["Class"] = 0
            df_train.fillna(0, inplace=True)
            X = df_train.drop(columns=["Class"])
            y = df_train["Class"]
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_frac_sidebar), stratify=y, random_state=42)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_frac_sidebar), random_state=42)
            models_trained = train_models(X_train, y_train, rf_n=int(rf_n_sidebar))
            st.session_state["models"] = models_trained
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.success("All models trained and stored in session (via one-click).")
            st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset options")
use_default = st.sidebar.checkbox("Use default dataset (provided path)", value=True)
alt_path_input = st.sidebar.text_input("Alternate dataset path (optional)", value=str(DEFAULT_DATASET_PATH))
uploaded_file = st.sidebar.file_uploader("Or upload creditcard.csv (zip or csv)", type=["csv", "zip"])

# ------------------ Dataset loading UI ------------------
df = None
if uploaded_file is not None:
    df = load_from_uploaded(uploaded_file)
    if df is None:
        st.sidebar.error("Failed to read uploaded file.")
    else:
        st.sidebar.success("Uploaded dataset loaded")
        st.session_state["df_loaded"] = df

# Try alt path input if set and user wants default
if df is None and use_default:
    candidate = alt_path_input.strip() if alt_path_input and alt_path_input.strip() != "" else str(DEFAULT_DATASET_PATH)
    df = try_load_from_path(candidate)
    if df is None:
        df = try_load_from_path("creditcard.csv")
    if df is not None:
        st.sidebar.success(f"Loaded dataset from path: {candidate}")
        st.session_state["df_loaded"] = df

# If still none, fallback to synthetic dataset (non-failing)
if df is None:
    st.sidebar.warning("No dataset found at upload/default path. Using synthetic dataset for demo.")
    df = generate_synthetic()
    st.sidebar.info("Synthetic dataset generated (for demo only).")
    st.session_state["df_loaded"] = df

# Minimal validation & preprocessing
if "Class" not in df.columns:
    st.warning("Loaded dataset does not contain 'Class' column; adding synthetic target for demo.")
    df["Class"] = 0

df = df.copy()
df.fillna(0, inplace=True)
df["Class"] = df["Class"].astype(int)

# store canonical feature columns for batch scoring and custom input
feature_columns = ensure_session_columns(df)
st.session_state["feature_columns"] = feature_columns

# quick existence checks for Time/Amount
has_time = "Time" in df.columns
has_amount = "Amount" in df.columns

# Provide selectbox for navigation as well (keeps previous UX)
page = st.sidebar.selectbox("Choose page", pages, index=st.session_state["page_index"])
# keep session state consistent if selectbox changes
if page != st.session_state["selected_page"]:
    st.session_state["selected_page"] = page
    st.session_state["page_index"] = pages.index(page)

# ------------------ Pages Implementation ------------------

# Helper to get models existence
def models_ready():
    return "models" in st.session_state and st.session_state["models"] is not None

# 1. Dataset Upload & Preview
if st.session_state["selected_page"] == "1. Dataset Upload & Preview":
    st.header("1. Dataset Upload & Preview")
    safe_st_lottie("upload", height=240)
    st.subheader("Preview (first 10 rows)")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.markdown(
        """
    **6 Insights:**
    - Use a properly exported CSV/zip to avoid parsing issues.
    - Confirm 'Class' target exists (0 legitimate, 1 fraud).
    - Check for missing values and unexpected dtypes.
    - For production, store dataset snapshots for reproducibility.
    - Prefer column ordering consistent with training for batch scoring.
    - Mask or remove any PII before uploading to dashboards.
    """
    )

# 2. Dataset Summary & Preview
elif st.session_state["selected_page"] == "2. Dataset Summary & Preview":
    st.header("2. Dataset Summary & Preview")
    safe_st_lottie("dataset_preview", height=220)
    st.subheader("Basic summary")
    st.write(df.describe(include='all').T)
    st.subheader("Class distribution")
    try:
        st.bar_chart(df["Class"].value_counts())
    except Exception:
        st.write(df["Class"].value_counts())
    st.markdown(
        """
    **6 Insights:**
    - Class imbalance often extreme — handle carefully in training.
    - Statistical summaries reveal skew, outliers, and scale issues.
    - Inspect top unique values for categorical leaks.
    - Use downsampling/upsampling or class weights for training.
    - Keep a holdout set for unbiased evaluation.
    - Add logging when dataset changes (schema drift detection).
    """
    )

# 3. Univariate Analysis
elif st.session_state["selected_page"] == "3. Univariate Analysis":
    st.header("3. Univariate Analysis")
    safe_st_lottie("bar_chart", height=170)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Class", data=df, ax=ax)
        ax.set_title("Class Distribution (0=Normal,1=Fraud)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if has_amount:
            sns.histplot(df["Amount"], bins=80, kde=True, ax=ax)
            ax.set_title("Transaction Amount Distribution")
        else:
            ax.text(0.5, 0.5, "No 'Amount' column", ha="center")
        st.pyplot(fig)
    st.markdown(
        """
    **6 Insights:**
    1. Fraud cases are rare compared to normal transactions.
    2. Amount distribution is usually right-skewed with a heavy tail.
    3. Many transactions are low-value; a few are very large.
    4. Consider log-transform or robust scaling for Amount.
    5. Visual checks help identify features for rules.
    6. Prepare sampling/weighting strategy for model training.
    """
    )

# 4. Bivariate Analysis
elif st.session_state["selected_page"] == "4. Bivariate Analysis":
    st.header("4. Bivariate Analysis")
    safe_st_lottie("scatter", height=170)
    sampleN = min(20000, len(df))
    sample = df.sample(sampleN, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    if has_amount:
        sns.boxplot(x="Class", y="Amount", data=sample, ax=ax)
        ax.set_title("Amount by Class (sample)")
    else:
        ax.text(0.5, 0.5, "No 'Amount' column", ha="center")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if has_time:
        sns.boxplot(x="Class", y="Time", data=sample, ax=ax2)
        ax2.set_title("Time by Class (sample)")
    else:
        ax2.text(0.5, 0.5, "No 'Time' column", ha="center")
    st.pyplot(fig2)

    st.markdown(
        """
    **6 Insights:**
    - Boxplots reveal distributional differences across classes.
    - Amount alone is often insufficient to detect fraud.
    - Temporal patterns (Time) can indicate fraud windows.
    - Combine multiple features for stronger signals.
    - Visualize pairwise interactions for feature design.
    - Use these observations to craft velocity/session features.
    """
    )

# 5. Multivariate Analysis
elif st.session_state["selected_page"] == "5. Multivariate Analysis":
    st.header("5. Multivariate Analysis")
    safe_st_lottie("heatmap", height=170)
    sample = df.sample(min(10000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = sample.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (sample)")
    st.pyplot(fig)

    if st.button("Show PCA 2D projection (sample)"):
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        Xs = scaler.fit_transform(sample[feature_columns])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=sample["Class"], palette="coolwarm", alpha=0.6)
        ax.set_title("PCA 2D Projection")
        st.pyplot(fig)
    st.markdown(
        """
    **6 Insights:**
    1. Correlations highlight related features and weak predictors.
    2. PCA provides a compact view for clustering and visualization.
    3. Multivariate anomalies can indicate coordinated fraud.
    4. Use embeddings or PCA for analyst dashboards.
    5. Feature engineering (time/aggregation) increases signal.
    6. Drop redundant features to speed production inference.
    """
    )

# 6. Outlier Detection
elif st.session_state["selected_page"] == "6. Outlier Detection":
    st.header("6. Outlier Detection")
    safe_st_lottie("alert", height=170)
    cont = st.slider("IsolationForest contamination", 0.001, 0.05, 0.01, step=0.001)
    sampleN = min(20000, len(df))
    sample = df.sample(sampleN, random_state=42)
    if has_amount and has_time:
        iso = IsolationForest(contamination=float(cont), random_state=42)
        try:
            iso.fit(sample[["Amount", "Time"]])
            preds = iso.predict(sample[["Amount", "Time"]])
            sample = sample.copy()
            sample["outlier"] = (preds == -1).astype(int)
            fig, ax = plt.subplots(figsize=(8, 6))
            palette_map = {0: "blue", 1: "red"}
            sns.scatterplot(x="Amount", y="Time", hue="outlier", data=sample, palette=palette_map, alpha=0.6, ax=ax)
            ax.set_title("Outliers (IsolationForest) — Amount vs Time (sample)")
            st.pyplot(fig)
            st.write("Outliers detected:", int(sample["outlier"].sum()))
        except Exception as e:
            st.error(f"Outlier detection failed: {e}")
    else:
        st.info("Dataset missing 'Amount' or 'Time' columns required for this demo.")
    st.markdown(
        """
    **6 Insights:**
    - IsolationForest flags anomalous transactions in feature space.
    - Outliers may be fraud or rare, legitimate events.
    - Outlier score can be added as a model feature.
    - Don't remove outliers blindly; they may be fraud.
    - Visual inspection should precede filtering.
    - Use outlier detection for triage and analyst prioritization.
    """
    )

# 7. Model Training
elif st.session_state["selected_page"] == "7. Model Training":
    st.header("7. Model Training")
    safe_st_lottie("training", height=200)
    test_frac = st.slider("Test set fraction", 0.1, 0.5, 0.3, key="train_test_frac")
    rf_n = st.slider("RandomForest n_estimators", 50, 500, 100, step=10, key="train_rf_n")
    if st.button("Train models (RandomForest, LogisticRegression, GradientBoosting)"):
        X = df.drop(columns=["Class"])
        y = df["Class"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_frac), stratify=y, random_state=42)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_frac), random_state=42)
        models = train_models(X_train, y_train, rf_n=rf_n)
        st.session_state["models"] = models
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.success("Models trained and stored in session.")
    st.markdown(
        """
    **6 Insights:**
    1. We train multiple models to compare performance and latency.
    2. Use class_weight or resampling to address imbalance.
    3. Save trained models for reproducible deployments.
    4. Monitor model performance for drift and retrain as needed.
    5. For real-time, prefer low-latency model serving for sub-second decisions.
    6. Use a feature store for consistent offline/online features.
    """
    )

# 8. Confusion Matrix
elif st.session_state["selected_page"] == "8. Confusion Matrix":
    st.header("8. Confusion Matrix")
    safe_st_lottie("matrix", height=160)
    if not models_ready():
        st.warning('Train models first in "7. Model Training" or use the sidebar Train All Models.')
    else:
        models = st.session_state["models"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        sel = st.selectbox("Choose model", list(models.keys()), key="cm_model_select")
        model = models[sel]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {sel}")
        st.pyplot(fig)
        st.markdown(
            """
        **6 Insights:**
        - Confusion matrix shows TP, FP, FN, TN counts.
        - In fraud detection, FN (missed fraud) is high cost.
        - Use cost-based thresholds to pick operating point.
        - Adjust thresholds or add manual review for uncertain cases.
        - Monitor confusion matrix over time for drift.
        - Combine automated and human review to reduce customer friction.
        """
        )

# 9. ROC & PR Curves
elif st.session_state["selected_page"] == "9. ROC & PR Curves":
    st.header("9. ROC & Precision-Recall Curves")
    safe_st_lottie("roc", height=160)
    if not models_ready():
        st.warning("Train models first.")
    else:
        models = st.session_state["models"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, m in models.items():
            y_scores = safe_predict_proba(m, X_test)
            try:
                prec, rec, _ = precision_recall_curve(y_test, y_scores)
                pr_auc = auc(rec, prec)
                ax.plot(rec, prec, label=f"{name} PR-AUC={pr_auc:.3f}")
            except Exception:
                continue
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend()
        st.pyplot(fig)
        st.markdown(
            """
        **6 Insights:**
        1. Precision-Recall is critical for imbalanced datasets.
        2. Choose operating point by business cost tradeoffs.
        3. ROC-AUC can be optimistic on imbalanced data.
        4. Calibrating probabilities may improve decisions.
        5. Compare models using PR-AUC and ROC-AUC.
        6. Continuously monitor curves to detect degradation.
        """
        )

# 10. Feature Importance
elif st.session_state["selected_page"] == "10. Feature Importance":
    st.header("10. Feature Importance (RandomForest)")
    safe_st_lottie("feature", height=170)
    if not models_ready():
        st.warning("Train models first.")
    else:
        rf = st.session_state["models"].get("RandomForest")
        X_test = st.session_state["X_test"]
        if rf is None:
            st.error("RandomForest not found in session models.")
        else:
            fi = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)[:30]
            fig, ax = plt.subplots(figsize=(8, 10))
            sns.barplot(x=fi.values, y=fi.index, ax=ax)
            ax.set_title("Top Feature Importances (RandomForest)")
            st.pyplot(fig)
            st.markdown(
                """
            **6 Insights:**
            - Feature importances provide a global view of model drivers.
            - Use importance to prioritize monitoring of features.
            - Combine with SHAP for per-prediction explanations.
            - Low-importance features can be removed to speed inference.
            - Re-evaluate importances after retraining.
            - Importance varies by model and dataset slice.
            """
            )

# 11. SHAP Explainability
elif st.session_state["selected_page"] == "11. SHAP Explainability":
    st.header("11. SHAP Explainability")
    safe_st_lottie("shap", height=170)
    if not SHAP_AVAILABLE:
        st.error("SHAP library is not installed. Install `shap` to use this feature.")
    elif not models_ready():
        st.warning("Train models first.")
    else:
        rf = st.session_state["models"].get("RandomForest")
        X_test = st.session_state["X_test"]
        sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(sample)
            fig = plt.figure(figsize=(8, 6))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap.summary_plot(shap_values[1], sample, plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
        st.markdown(
            """
        **6 Insights:**
        - SHAP explains contribution of each feature to predictions.
        - Use class-specific SHAP (index 1) for fraud explanations if available.
        - Local explanations are useful for investigator UI.
        - Aggregate SHAP gives global feature behavior.
        - Keep SHAP sample sizes moderate for dashboard responsiveness.
        - Combine SHAP with rules for auditable decisions.
        """
        )

# 12. Model Comparison
elif st.session_state["selected_page"] == "12. Model Comparison":
    st.header("12. Model Comparison")
    safe_st_lottie("compare", height=160)
    if not models_ready():
        st.warning("Train models first.")
    else:
        models = st.session_state["models"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        res = []
        for name, m in models.items():
            scores = safe_predict_proba(m, X_test)
            try:
                roc = roc_auc_score(y_test, scores)
            except Exception:
                roc = float("nan")
            res.append({"model": name, "roc_auc": roc})
        st.table(pd.DataFrame(res))
        st.markdown(
            """
        **6 Insights:**
        - Compare models on ROC-AUC/PR-AUC to choose baseline.
        - Consider latency and explainability when picking production model.
        - Ensembles can boost performance but increase complexity.
        - Validate models on recent data (temporal split).
        - Keep model registry and metadata for reproducibility.
        - Evaluate cost-benefit for stricter or looser policies.
        """
        )

# 13. Real-Time Simulation
elif st.session_state["selected_page"] == "13. Real-Time Simulation":
    st.header("13. Real-Time Simulation")
    safe_st_lottie("live", height=160)
    if not models_ready():
        st.warning("Train models first.")
    else:
        rf = st.session_state["models"].get("RandomForest")
        X = df.drop(columns=["Class"])
        num = st.number_input("Number of events to stream", 1, 500, 50)
        delay = st.number_input("Delay between events (seconds)", 0.0, 5.0, 0.1, step=0.1)
        stream_button = st.button("Start streaming simulation")
        if stream_button:
            sample = X.sample(int(num), random_state=np.random.randint(0, 10000))
            placeholder = st.empty()
            progress = st.progress(0)
            total = len(sample)
            for i, (idx, row) in enumerate(sample.iterrows(), start=1):
                score = safe_predict_proba(rf, row.values.reshape(1, -1))[0]
                decision = "⚠️ FRAUD" if score > 0.5 else "✅ LEGIT"
                placeholder.write(f"Transaction {i} | Score: {score:.4f} | Decision: {decision}")
                progress.progress(int(i / total * 100))
                time.sleep(min(float(delay), 1.0))
            placeholder.write("Streaming finished.")
        st.markdown(
            """
        **6 Insights:**
        - Streaming simulates ingestion and scoring pipeline.
        - For production, replace sample() with Kafka/Kinesis ingestion.
        - Use low-latency model serving for sub-second decisioning.
        - Rate-limit alerts to prevent operator overload.
        - Persist decisions for audit and retraining.
        - Provide analyst UI for triage of streamed alerts.
        """
        )

# 14. Custom Transaction Input
elif st.session_state["selected_page"] == "14. Custom Transaction Input":
    st.header("14. Custom Transaction Input")
    safe_st_lottie("input", height=160)
    cols = st.columns(3)
    with cols[0]:
        time_v = st.number_input("Time", value=int(df["Time"].median()) if has_time else 0)
        amount_v = st.number_input("Amount", value=float(df["Amount"].median()) if has_amount else 0.0)
    st.markdown("Enter PCA features V1..V28 as comma-separated values (or leave zeros):")
    default_v = ",".join(["0"] * 28)
    v_input = st.text_area("V1..V28", value=default_v, height=120)
    if st.button("Score custom transaction"):
        try:
            v_vals = [float(x.strip()) for x in v_input.split(",") if x.strip() != ""]
            if len(v_vals) != 28:
                st.error("Please enter exactly 28 values for V1..V28.")
            else:
                cols_order = st.session_state.get("feature_columns", list(df.drop(columns=["Class"]).columns))
                row_dict = {}
                for c in cols_order:
                    if c == "Time":
                        row_dict[c] = time_v
                    elif c == "Amount":
                        row_dict[c] = amount_v
                    elif c.startswith("V"):
                        idx = int(c[1:]) - 1
                        row_dict[c] = v_vals[idx] if 0 <= idx < 28 else 0.0
                    else:
                        row_dict[c] = 0.0
                row_df = pd.DataFrame([row_dict], columns=cols_order)
                if not models_ready():
                    st.warning("Train models first.")
                else:
                    rf = st.session_state["models"].get("RandomForest")
                    score = safe_predict_proba(rf, row_df)[0]
                    decision = "⚠️ FRAUD" if score > 0.5 else "✅ LEGIT"
                    st.write(f"Score: {score:.4f} | Decision: {decision}")
        except Exception as e:
            st.error(f"Failed to parse input: {e}")
    st.markdown(
        """
    **6 Insights:**
    - Useful for QA, incident reproduction, and analyst testing.
    - Must follow same preprocessing and feature ordering as training.
    - Store interesting inputs to enrich training data later.
    - Ensure input validation for production usage.
    - Provide SHAP snapshot for flagged custom inputs.
    - Integrate with case management for further investigation.
    """
    )

# 15. Batch Prediction
elif st.session_state["selected_page"] == "15. Batch Prediction":
    st.header("15. Batch Prediction")
    safe_st_lottie("batch", height=160)
    uploaded_batch = st.file_uploader("Upload CSV (columns matching training features, no Class) for batch scoring", type=["csv"], key="batch_uploader")
    if uploaded_batch is not None and models_ready():
        try:
            uploaded_batch.seek(0)
            batch = pd.read_csv(uploaded_batch)
            cols_order = st.session_state.get("feature_columns", list(df.drop(columns=["Class"]).columns))
            missing = [c for c in cols_order if c not in batch.columns]
            if missing:
                st.error(f"Uploaded batch is missing columns required for scoring: {missing}")
            else:
                batch_aligned = batch[cols_order].copy()
                rf = st.session_state["models"].get("RandomForest")
                scores = safe_predict_proba(rf, batch_aligned)
                batch["score"] = scores
                st.dataframe(batch.head())
                csv = batch.to_csv(index=False).encode("utf-8")
                st.download_button("Download scored CSV", csv, "scored_batch.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")
    elif uploaded_batch is not None and not models_ready():
        st.warning("Train models first.")
    st.markdown(
        """
    **6 Insights:**
    - Batch scoring is suitable for offline workflows/nightly jobs.
    - Ensure column alignment and identical preprocessing as training.
    - Use batch scoring to generate labels for analyst review.
    - For large batches, use chunking and background workers.
    - Persist batch outputs to data warehouse for analytics.
    - Monitor batch latency and failures.
    """
    )

# 16. Download Predictions
elif st.session_state["selected_page"] == "16. Download Predictions":
    st.header("16. Download Predictions")
    safe_st_lottie("download", height=150)
    if models_ready():
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        rf = st.session_state["models"].get("RandomForest")
        scores = safe_predict_proba(rf, X_test)
        out = X_test.copy()
        out["true_class"] = y_test.values
        out["score"] = scores
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, "predictions.csv", mime="text/csv")
    else:
        st.warning("Train models first to generate predictions.")
    st.markdown(
        """
    **6 Insights:**
    - Export predictions for auditing and investigative workflows.
    - Include context and SHAP snapshots for high-risk cases.
    - Protect and encrypt exported files for compliance.
    - Automate recurring exports to analysts or case systems.
    - Track export history for reproducibility.
    - Apply retention policies according to regulations.
    """
    )

# 17. Alerts Simulation
elif st.session_state["selected_page"] == "17. Alerts Simulation":
    st.header("17. Alerts Simulation")
    safe_st_lottie("notify", height=150)
    if not models_ready():
        st.warning("Train models first.")
    else:
        rf = st.session_state["models"].get("RandomForest")
        X = df.drop(columns=["Class"])
        alert_thresh = st.slider("Alert threshold (score >= triggers alert)", 0.01, 0.99, 0.6, step=0.01, key="alert_thresh")
        n = st.number_input("Number of events to scan", 1, 1000, 100, key="alert_n")
        if st.button("Run alert simulation"):
            sample = X.sample(int(n), random_state=np.random.randint(0, 10000))
            alerts = []
            for idx, row in sample.iterrows():
                score = safe_predict_proba(rf, row.values.reshape(1, -1))[0]
                if score >= alert_thresh:
                    alerts.append({"id": int(idx), "score": float(score)})
            st.write(f"Alerts generated: {len(alerts)}")
            st.json(alerts[:50])
    st.markdown(
        """
    **6 Insights:**
    - Alerts simulate automated notifications to analysts/ops.
    - Tune thresholds to balance workload vs coverage.
    - Add rate-limiting, batching, and deduplication to alert flows.
    - Include SHAP and context in alert payloads for faster triage.
    - Integrate with Slack/PagerDuty for incident management.
    - Measure alert-to-resolution time as an ops KPI.
    """
    )

# 18. Model Save/Load
elif st.session_state["selected_page"] == "18. Model Save/Load":
    st.header("18. Model Save & Load")
    safe_st_lottie("save", height=150)
    if models_ready() and st.session_state["models"].get("RandomForest") is not None:
        if st.button("Save RandomForest to file (rf_model.joblib)"):
            try:
                joblib.dump(st.session_state["models"]["RandomForest"], "rf_model.joblib")
                st.success("Saved rf_model.joblib to working directory.")
                with open("rf_model.joblib", "rb") as f:
                    st.download_button("Download saved rf_model.joblib", f, file_name="rf_model.joblib")
            except Exception as e:
                st.error(f"Failed to save model: {e}")
    uploaded_model = st.file_uploader("Or upload a joblib model to load (as RandomForest)", type=["joblib", "pkl"], key="model_uploader")
    if uploaded_model is not None:
        try:
            uploaded_model.seek(0)
            loaded = joblib.load(uploaded_model)
            st.session_state["models"] = {"RandomForest": loaded}
            st.success("Model loaded into session as RandomForest")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    st.markdown(
        """
    **6 Insights:**
    - Save model artifacts for reproducible deployments.
    - Use a model registry (MLflow, Sagemaker, Seldon) for production.
    - Validate models on a holdout before promoting to production.
    - Version models and store metadata (data snapshot, params).
    - Use canary and blue/green deployments for safe rollouts.
    - Secure model artifacts and limit access.
    """
    )

# Footer note
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Demo only: for production use streaming ingestion (Kafka), an online feature store (Feast/Redis), and secure model-serving. Ensure PCI/GDPR compliance."
)
