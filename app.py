"""
European Options Implied Volatility Calculator
Machine Learning-based IV prediction using Random Forest, XGBoost, and Neural Networks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
import time
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="European Options IV Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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


# Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def vega(S, K, T, r, sigma):
    """Calculate vega."""
    if T <= 0:
        return 0.0001

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega_value = S * norm.pdf(d1) * np.sqrt(T)

    return max(vega_value, 0.0001)


def implied_volatility_newton(option_price, S, K, T, r, option_type='call', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson method."""
    sigma = 0.2  # Initial guess

    for i in range(max_iterations):
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)

        vega_val = vega(S, K, T, r, sigma)
        diff = option_price - price

        if abs(diff) < 1e-6:
            return sigma

        sigma = sigma + diff / vega_val
        sigma = max(0.001, min(sigma, 5.0))

    return sigma


# Model training function (cached)
@st.cache_resource
def load_or_train_models():
    """Load pre-trained models or train new ones."""
    st.info("Loading models... This may take a moment on first run.")

    # Check if models exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    rf_path = os.path.join(model_dir, "rf_model.pkl")
    xgb_path = os.path.join(model_dir, "xgb_model.pkl")

    try:
        # Try to load existing models
        if os.path.exists(rf_path) and os.path.exists(xgb_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            with open(xgb_path, 'rb') as f:
                xgb_model = pickle.load(f)
            st.success("Pre-trained models loaded successfully!")
            return rf_model, xgb_model
    except:
        pass

    # Train new models if not found
    st.warning("Pre-trained models not found. Please train models first using the notebooks.")
    st.info("For demo purposes, using placeholder models with limited accuracy.")

    # Create simple demo models
    rf_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=10, max_depth=5, random_state=42)

    # Generate some synthetic training data for demo
    np.random.seed(42)
    n_samples = 1000
    X_demo = pd.DataFrame({
        'T_years': np.random.uniform(0.01, 2, n_samples),
        'moneyness': np.random.uniform(0.8, 1.2, n_samples),
        'risk_free_rate': np.random.uniform(0.01, 0.05, n_samples)
    })
    y_demo = 0.2 + 0.1 * (X_demo['moneyness'] - 1) + 0.05 * X_demo['T_years'] + np.random.normal(0, 0.02, n_samples)

    rf_model.fit(X_demo, y_demo)
    xgb_model.fit(X_demo, y_demo)

    return rf_model, xgb_model


# Main app
def main():
    st.title("📈 European Options Implied Volatility Calculator")
    st.markdown("### Machine Learning-based IV Prediction")

    # Load models
    rf_model, xgb_model = load_or_train_models()

    # Sidebar inputs
    st.sidebar.header("Option Parameters")

    underlying_price = st.sidebar.number_input(
        "Underlying Price (S)",
        min_value=1.0,
        max_value=10000.0,
        value=4000.0,
        step=10.0,
        help="Current price of the underlying asset (e.g., SPX index)"
    )

    strike_price = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=1.0,
        max_value=10000.0,
        value=4100.0,
        step=10.0,
        help="Strike price of the option"
    )

    days_to_expiration = st.sidebar.number_input(
        "Days to Expiration",
        min_value=1,
        max_value=730,
        value=30,
        step=1,
        help="Number of days until option expiration"
    )

    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
        help="Annual risk-free interest rate"
    ) / 100

    option_type = st.sidebar.selectbox(
        "Option Type",
        ["Call", "Put"],
        help="Type of option"
    )

    # Calculate derived features
    T_years = days_to_expiration / 365.0
    moneyness = underlying_price / strike_price

    # Display calculated features
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Calculated Features**")
    st.sidebar.metric("Time to Expiration (years)", f"{T_years:.4f}")
    st.sidebar.metric("Moneyness (S/K)", f"{moneyness:.4f}")
    st.sidebar.metric("ITM/OTM", "ITM" if (moneyness > 1 and option_type == "Call") or (moneyness < 1 and option_type == "Put") else "OTM")

    # Prediction button
    if st.sidebar.button("Calculate IV", type="primary"):
        # Prepare input
        X_input = pd.DataFrame({
            'T_years': [T_years],
            'moneyness': [moneyness],
            'risk_free_rate': [risk_free_rate]
        })

        # Create columns for results
        col1, col2, col3 = st.columns(3)

        # Random Forest prediction
        with col1:
            st.markdown("### 🌲 Random Forest")
            start_time = time.time()
            rf_iv = rf_model.predict(X_input)[0]
            rf_time = time.time() - start_time

            st.metric("Implied Volatility", f"{rf_iv*100:.2f}%")
            st.metric("Calculation Time", f"{rf_time*1000:.2f} ms")

            # Calculate option price
            if option_type == "Call":
                rf_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, rf_iv)
            else:
                rf_price = black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, rf_iv)

            st.metric("Estimated Option Price", f"${rf_price:.2f}")

        # XGBoost prediction
        with col2:
            st.markdown("### 🚀 XGBoost")
            start_time = time.time()
            xgb_iv = xgb_model.predict(X_input)[0]
            xgb_time = time.time() - start_time

            st.metric("Implied Volatility", f"{xgb_iv*100:.2f}%")
            st.metric("Calculation Time", f"{xgb_time*1000:.2f} ms")

            if option_type == "Call":
                xgb_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, xgb_iv)
            else:
                xgb_price = black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, xgb_iv)

            st.metric("Estimated Option Price", f"${xgb_price:.2f}")

        # Black-Scholes comparison (given an assumed IV)
        with col3:
            st.markdown("### 📊 Black-Scholes")
            st.info("Using ML average IV")

            avg_iv = (rf_iv + xgb_iv) / 2

            if option_type == "Call":
                bs_price = black_scholes_call(underlying_price, strike_price, T_years, risk_free_rate, avg_iv)
            else:
                bs_price = black_scholes_put(underlying_price, strike_price, T_years, risk_free_rate, avg_iv)

            st.metric("Implied Volatility", f"{avg_iv*100:.2f}%")
            st.metric("Option Price", f"${bs_price:.2f}")

        # Visualization section
        st.markdown("---")
        st.markdown("### 📈 Volatility Surface & Analysis")

        tab1, tab2, tab3 = st.tabs(["Volatility Surface", "Price Sensitivity", "Model Comparison"])

        with tab1:
            # Generate IV surface
            strikes = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 20)
            expiries = np.linspace(1/365, 365/365, 20)

            strike_mesh, expiry_mesh = np.meshgrid(strikes, expiries)
            moneyness_mesh = underlying_price / strike_mesh

            iv_surface = np.zeros_like(strike_mesh)

            for i in range(len(expiries)):
                for j in range(len(strikes)):
                    X_surf = pd.DataFrame({
                        'T_years': [expiry_mesh[i, j]],
                        'moneyness': [moneyness_mesh[i, j]],
                        'risk_free_rate': [risk_free_rate]
                    })
                    iv_surface[i, j] = rf_model.predict(X_surf)[0]

            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(
                x=strike_mesh,
                y=expiry_mesh,
                z=iv_surface * 100,
                colorscale='Viridis',
                colorbar=dict(title="IV (%)")
            )])

            fig.update_layout(
                title="Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Strike Price",
                    yaxis_title="Time to Expiration (years)",
                    zaxis_title="Implied Volatility (%)"
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Greeks and sensitivity analysis
            st.markdown("#### Option Price Sensitivity")

            # Vary underlying price
            S_range = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 50)
            prices_call = []
            prices_put = []

            for S in S_range:
                X_sens = pd.DataFrame({
                    'T_years': [T_years],
                    'moneyness': [S / strike_price],
                    'risk_free_rate': [risk_free_rate]
                })
                iv_pred = rf_model.predict(X_sens)[0]

                prices_call.append(black_scholes_call(S, strike_price, T_years, risk_free_rate, iv_pred))
                prices_put.append(black_scholes_put(S, strike_price, T_years, risk_free_rate, iv_pred))

            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=S_range, y=prices_call, mode='lines', name='Call', line=dict(color='green', width=3)))
            fig.add_trace(go.Scatter(x=S_range, y=prices_put, mode='lines', name='Put', line=dict(color='red', width=3)))
            fig.add_vline(x=underlying_price, line_dash="dash", line_color="gray", annotation_text="Current Price")
            fig.add_vline(x=strike_price, line_dash="dash", line_color="blue", annotation_text="Strike")

            fig.update_layout(
                title="Option Price vs Underlying Price",
                xaxis_title="Underlying Price ($)",
                yaxis_title="Option Price ($)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Model comparison
            st.markdown("#### Model Performance Comparison")

            comparison_data = {
                'Model': ['Random Forest', 'XGBoost'],
                'IV Prediction (%)': [rf_iv * 100, xgb_iv * 100],
                'Calculation Time (ms)': [rf_time * 1000, xgb_time * 1000]
            }

            comparison_df = pd.DataFrame(comparison_data)

            col_a, col_b = st.columns(2)

            with col_a:
                fig = px.bar(comparison_df, x='Model', y='IV Prediction (%)',
                            title='IV Predictions', color='Model')
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig = px.bar(comparison_df, x='Model', y='Calculation Time (ms)',
                            title='Speed Comparison', color='Model')
                st.plotly_chart(fig, use_container_width=True)

    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")

    with st.expander("How It Works"):
        st.markdown("""
        This application uses machine learning models trained on historical SPX options data to predict implied volatility:

        - **Random Forest**: Ensemble of decision trees, excellent for feature importance analysis
        - **XGBoost**: Gradient boosting algorithm, optimized for speed and accuracy
        - **Neural Networks**: Deep learning approach for capturing complex patterns (available in notebooks)

        The models are trained on features including:
        - Time to expiration (years)
        - Moneyness (S/K ratio)
        - Risk-free rate
        """)

    with st.expander("Model Training"):
        st.markdown("""
        Models are trained on SPX European options data with the following characteristics:

        - **Dataset**: ~2.8M option contracts from 2022
        - **Train/Val/Test Split**: 70/15/15 (random split)
        - **Performance**: R² ≈ 0.40-0.45 on test set
        - **Speed**: 100-1000x faster than numerical Black-Scholes solver

        See the Jupyter notebooks in the `Notebooks/` directory for full training details.
        """)

    with st.expander("Limitations & Disclaimers"):
        st.markdown("""
        ⚠️ **Important Disclaimers**:

        - This tool is for educational and research purposes only
        - Not financial advice - do not use for actual trading decisions
        - Model predictions are based on historical data and may not reflect current market conditions
        - Always consult with a qualified financial advisor before making investment decisions
        - The models may not account for all market factors (volatility smile, term structure, etc.)
        """)


if __name__ == "__main__":
    main()
