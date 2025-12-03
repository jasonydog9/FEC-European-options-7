"""
European Options Implied Volatility Calculator
Machine Learning-based IV prediction using Random Forest and XGBoost
Trains on CSVs found in data/model_input/*.csv (expects columns like STRIKE, moneyness, T_years, iv, risk_free_rate)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import norm
import time
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# -------------------
# Page config / CSS
# -------------------
st.set_page_config(
    page_title="European Options IV Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------
# Black-Scholes utils
# -------------------
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0001
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega_value = S * norm.pdf(d1) * np.sqrt(T)
    return max(vega_value, 0.0001)

def implied_volatility_newton(option_price, S, K, T, r, option_type='call', max_iterations=100):
    sigma = 0.2
    for _ in range(max_iterations):
        price = black_scholes_call(S, K, T, r, sigma) if option_type == 'call' else black_scholes_put(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        diff = option_price - price
        if abs(diff) < 1e-6:
            return sigma
        sigma = sigma + diff / v
        sigma = max(0.001, min(sigma, 5.0))
    return sigma

# -------------------
# Paths
# -------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = MODELS_DIR / "rf_model.pkl"
XGB_PATH = MODELS_DIR / "xgb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURE_COLS = ["T_years", "moneyness", "risk_free_rate"]
TARGET_COL = "iv"  # as in your CSVs

# -------------------
# Load / Train Models
# -------------------
@st.cache_resource
def load_or_train_models():
    """
    Loads models and scaler if saved; otherwise trains on CSVs in data/model_input.
    Returns (rf_model, xgb_model, scaler).
    """

    # Try loading saved models
    if RF_PATH.exists() and XGB_PATH.exists() and SCALER_PATH.exists():
        try:
            rf_model = joblib.load(RF_PATH)
            xgb_model = joblib.load(XGB_PATH)
            scaler = joblib.load(SCALER_PATH)
            st.success("📦 Loaded pre-trained models & scaler from disk.")
            return rf_model, xgb_model, scaler
        except Exception as e:
            st.warning(f"Failed loading saved models: {e}. Re-training...")

    # -----------------------------------------------------------
    # 1) Locate CSV files in data/model_input
    # -----------------------------------------------------------
    csv_files = list((DATA_DIR / "model_input").glob("*.csv"))

    if len(csv_files) == 0:
        st.warning("No CSV files found under data/model_input. Falling back to synthetic demo models.")
        return _train_demo_models()

    # -----------------------------------------------------------
    # 2) Load all CSVs
    # -----------------------------------------------------------
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")

    if len(df_list) == 0:
        st.error("No valid CSVs loaded. Falling back to demo models.")
        return _train_demo_models()

    data = pd.concat(df_list, ignore_index=True)
    st.info(f"📄 Loaded {len(csv_files)} files, {len(data):,} rows")
    st.write("Preview columns:", data.columns.tolist())

    # -----------------------------------------------------------
    # 3) Validate required columns
    # -----------------------------------------------------------
    # Your CSV schema:
    # ['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'option_type', 'iv',
    #  'iv_normalized', 'T_years', 'moneyness', 'risk_free_rate']

    required = {"STRIKE", "moneyness", "T_years", "iv", "risk_free_rate"}
    missing = [c for c in required if c not in data.columns]

    if len(missing) > 0:
        st.error(f"❌ Required columns missing from your CSVs: {missing}")
        st.stop()

    # -----------------------------------------------------------
    # 4) Clean + cast numeric
    # -----------------------------------------------------------
    num_cols = ["STRIKE", "moneyness", "T_years", "iv", "risk_free_rate"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    before = len(data)
    data = data.dropna(subset=num_cols)
    st.write(f"🧹 Dropped {before - len(data):,} rows with NaNs")

    # Filter out blatantly invalid rows
    data = data[(data["iv"] > 0) & (data["iv"] < 5)]
    data = data[data["T_years"] > 0]

    # -----------------------------------------------------------
    # 5) Derive underlying_price (S)
    # -----------------------------------------------------------
    data["underlying_price"] = data["moneyness"] * data["STRIKE"]

    # -----------------------------------------------------------
    # 6) Select features + target
    # -----------------------------------------------------------
    # Automatically detect feature columns
    FEATURE_COLS = ["T_years", "moneyness", "risk_free_rate", "underlying_price", "STRIKE"]

    missing_features = [c for c in FEATURE_COLS if c not in data.columns]
    if missing_features:
        st.error(f"❌ Missing features: {missing_features}")
        st.stop()

    X = data[FEATURE_COLS].copy()
    y = data["iv"].copy()

    st.info(f"📊 Training dataset: {len(X):,} rows, {len(FEATURE_COLS)} features")

    # -----------------------------------------------------------
    # 7) Scale inputs
    # -----------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------------
    # 8) Train models
    # -----------------------------------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

    with st.spinner("🌲 Training Random Forest..."):
        rf_model.fit(X_scaled, y)

    with st.spinner("🚀 Training XGBoost..."):
        xgb_model.fit(X_scaled, y, verbose=False)

    # -----------------------------------------------------------
    # 9) Evaluate (train-set quick check)
    # -----------------------------------------------------------
    preds_rf = rf_model.predict(X_scaled)
    preds_xgb = xgb_model.predict(X_scaled)

    st.success("🎯 Model Performance (train set)")

    st.write(f"RandomForest — R²: {r2_score(y, preds_rf):.4f}, RMSE: {mean_squared_error(y, preds_rf, squared=False):.6f}")
    st.write(f"XGBoost      — R²: {r2_score(y, preds_xgb):.4f}, RMSE: {mean_squared_error(y, preds_xgb, squared=False):.6f}")

    # -----------------------------------------------------------
    # 10) Save models + scaler
    # -----------------------------------------------------------
    try:
        joblib.dump(rf_model, RF_PATH)
        joblib.dump(xgb_model, XGB_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.info(f"💾 Saved models and scaler to {MODELS_DIR}")
    except Exception as e:
        st.warning(f"⚠️ Could not save models: {e}")

    return rf_model, xgb_model, scaler


def _train_demo_models():
    """Fallback synthetic demo training (keeps previous behavior)."""
    np.random.seed(42)
    n_samples = 20000
    X_demo = pd.DataFrame({
        'T_years': np.random.uniform(0.01, 2, n_samples),
        'moneyness': np.random.uniform(0.7, 1.3, n_samples),
        'risk_free_rate': np.random.uniform(0.01, 0.05, n_samples)
    })
    base_iv = 0.20
    smile_effect = 0.15 * np.abs(X_demo['moneyness'] - 1)
    term_effect = 0.05 / np.sqrt(X_demo['T_years'])
    noise = np.random.normal(0, 0.02, n_samples)
    y_demo = np.clip(base_iv + smile_effect + term_effect + noise, 0.05, 1.5)

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(X_demo)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)

    with st.spinner("Training demo Random Forest..."):
        rf_model.fit(X_demo_scaled, y_demo)
    with st.spinner("Training demo XGBoost..."):
        xgb_model.fit(X_demo_scaled, y_demo, verbose=False)

    # Save demo models for faster startup later
    try:
        joblib.dump(rf_model, RF_PATH)
        joblib.dump(xgb_model, XGB_PATH)
        joblib.dump(scaler, SCALER_PATH)
    except Exception:
        pass

    st.success("✅ Demo models trained.")
    return rf_model, xgb_model, scaler

# -------------------
# Main app
# -------------------
def main():
    st.title("📈 European Options Implied Volatility Calculator")
    st.markdown("### Machine Learning-based IV Prediction")

    rf_model, xgb_model, scaler = load_or_train_models()

    # Sidebar inputs
    st.sidebar.header("Option Parameters")

    underlying_price = st.sidebar.number_input(
        "Underlying Price (S)",
        min_value=1.0,
        max_value=100000.0,
        value=4000.0,
        step=1.0,
        help="Current price of the underlying asset (e.g., SPX index)"
    )

    strike_price = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=1.0,
        max_value=100000.0,
        value=4100.0,
        step=1.0,
        help="Strike price of the option"
    )

    days_to_expiration = st.sidebar.number_input(
        "Days to Expiration",
        min_value=1,
        max_value=3650,
        value=30,
        step=1
    )

    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.01
    ) / 100

    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    # Derived features
    T_years = days_to_expiration / 365.0
    moneyness = underlying_price / strike_price

    st.sidebar.markdown("---")
    st.sidebar.metric("Time to Expiration (years)", f"{T_years:.4f}")
    st.sidebar.metric("Moneyness (S/K)", f"{moneyness:.4f}")
    st.sidebar.metric("ITM/OTM", "ITM" if (moneyness > 1 and option_type == "Call") or (moneyness < 1 and option_type == "Put") else "OTM")

    # Prediction
    if st.sidebar.button("Calculate IV"):
        X_input_df = pd.DataFrame({
            "T_years": [T_years],
            "moneyness": [moneyness],
            "risk_free_rate": [risk_free_rate]
        })

        # scale input
        X_input_scaled = scaler.transform(X_input_df)

        col1, col2, col3 = st.columns(3)

        # Random Forest
        with col1:
            st.markdown("### 🌲 Random Forest")
            t0 = time.time()
            rf_iv_raw = rf_model.predict(X_input_scaled)[0]
            rf_iv = float(np.clip(rf_iv_raw, 0.0001, 5.0))
            rf_time = (time.time() - t0) * 1000.0
            if rf_iv_raw != rf_iv:
                st.warning(f"⚠️ Clipped from {rf_iv_raw:.4f} to valid range")
            st.metric("Implied Volatility", f"{rf_iv*100:.2f}%")
            st.metric("Calculation Time", f"{rf_time:.1f} ms")
            rf_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, rf_iv) if option_type == "Call" else black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, rf_iv)
            st.metric("Estimated Option Price", f"${rf_price:.2f}")

        # XGBoost
        with col2:
            st.markdown("### 🚀 XGBoost")
            t0 = time.time()
            xgb_iv_raw = xgb_model.predict(X_input_scaled)[0]
            xgb_iv = float(np.clip(xgb_iv_raw, 0.0001, 5.0))
            xgb_time = (time.time() - t0) * 1000.0
            if xgb_iv_raw != xgb_iv:
                st.warning(f"⚠️ Clipped from {xgb_iv_raw:.4f} to valid range")
            st.metric("Implied Volatility", f"{xgb_iv*100:.2f}%")
            st.metric("Calculation Time", f"{xgb_time:.1f} ms")
            xgb_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, xgb_iv) if option_type == "Call" else black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, xgb_iv)
            st.metric("Estimated Option Price", f"${xgb_price:.2f}")

        # Black-Scholes avg comparison
        with col3:
            st.markdown("### 📊 Black-Scholes (avg ML IV)")
            avg_iv = (rf_iv + xgb_iv) / 2.0
            bs_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, avg_iv) if option_type == "Call" else black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, avg_iv)
            st.metric("Implied Volatility", f"{avg_iv*100:.2f}%")
            st.metric("Option Price", f"${bs_price:.2f}")

        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Volatility Surface & Analysis")
        tab1, tab2, tab3 = st.tabs(["Volatility Surface", "Price Sensitivity", "Model Comparison"])

        with tab1:
            strikes = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 20)
            expiries = np.linspace(1/365, 365/365, 20)
            strike_mesh, expiry_mesh = np.meshgrid(strikes, expiries)
            moneyness_mesh = underlying_price / strike_mesh
            # Prepare feature grid and scale once
            grid_df = pd.DataFrame({
                "T_years": expiry_mesh.ravel(),
                "moneyness": moneyness_mesh.ravel(),
                "risk_free_rate": np.full(expiry_mesh.size, risk_free_rate)
            })
            grid_scaled = scaler.transform(grid_df[FEATURE_COLS])
            preds_grid = rf_model.predict(grid_scaled)
            iv_surface = np.clip(preds_grid.reshape(expiry_mesh.shape), 0.0001, 5.0) * 100.0

            fig = go.Figure(data=[go.Surface(
                x=strike_mesh,
                y=expiry_mesh,
                z=iv_surface,
                colorscale='Viridis',
                colorbar=dict(title="IV (%)")
            )])
            fig.update_layout(
                title="Implied Volatility Surface (Random Forest)",
                scene=dict(xaxis_title="Strike Price", yaxis_title="Time to Expiration (years)", zaxis_title="Implied Volatility (%)"),
                height=600
            )
            st.plotly_chart(fig, width='stretch')

        with tab2:
            st.markdown("#### Option Price Sensitivity")
            S_range = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 50)
            prices_call = []
            prices_put = []
            # vectorize predictions
            sens_df = pd.DataFrame({
                "T_years": np.full(S_range.size, T_years),
                "moneyness": S_range / strike_price,
                "risk_free_rate": np.full(S_range.size, risk_free_rate)
            })
            sens_scaled = scaler.transform(sens_df[FEATURE_COLS])
            iv_preds = np.clip(rf_model.predict(sens_scaled), 0.0001, 5.0)

            for i, S in enumerate(S_range):
                iv_pred = iv_preds[i]
                prices_call.append(black_scholes_call(S, strike_price, T_years, risk_free_rate, iv_pred))
                prices_put.append(black_scholes_put(S, strike_price, T_years, risk_free_rate, iv_pred))

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=S_range, y=prices_call, mode='lines', name='Call'))
            fig2.add_trace(go.Scatter(x=S_range, y=prices_put, mode='lines', name='Put'))
            fig2.add_vline(x=underlying_price, line_dash="dash", annotation_text="Current Price", line_color="gray")
            fig2.add_vline(x=strike_price, line_dash="dash", annotation_text="Strike", line_color="blue")
            fig2.update_layout(title="Option Price vs Underlying Price", xaxis_title="Underlying Price ($)", yaxis_title="Option Price ($)", height=500)
            st.plotly_chart(fig2, width='stretch')

        with tab3:
            st.markdown("#### Model Performance Comparison (single input)")
            comparison_df = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost"],
                "IV Prediction (%)": [rf_iv * 100.0, xgb_iv * 100.0],
                "Calculation Time (ms)": [rf_time, xgb_time]
            })
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(comparison_df, x='Model', y='IV Prediction (%)', title='IV Predictions', color='Model')
                st.plotly_chart(fig, width='stretch')
            with col_b:
                fig = px.bar(comparison_df, x='Model', y='Calculation Time (ms)', title='Speed Comparison', color='Model')
                st.plotly_chart(fig, width='stretch')

    # Info & disclaimers
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.info("Models are trained using CSVs in `data/model_input/`. Expected columns include: STRIKE, moneyness, T_years, iv, risk_free_rate. The app will derive underlying_price = moneyness * STRIKE.")
    with st.expander("Limitations & Disclaimers"):
        st.markdown("""
        ⚠️ For educational/research use only. Not financial advice.
        - Models trained on historical data and may not reflect current market conditions.
        - Consider a train/validation split and hyperparameter tuning for production.
        """)

if __name__ == "__main__":
    main()
