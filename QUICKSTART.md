# Quick Start Guide

## European Options Implied Volatility Calculator

This guide will help you get started with the project quickly.

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FEC-European-options-7
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Notebooks

### 1. Data Processing (Optional - data already processed)
If you have raw data files:
```bash
jupyter notebook Notebooks/data_processing.ipynb
```

### 2. Model Training
View the trained models and results:
```bash
jupyter notebook Notebooks/model_training.ipynb
```

### 3. Stretch Goal Analysis

**Black-Scholes Comparison:**
```bash
jupyter notebook Notebooks/stretch_goal_1_blackscholes.ipynb
```

**Training Set Size Analysis:**
```bash
jupyter notebook Notebooks/stretch_goal_2_training_size.ipynb
```

---

## Running the Web Application

### Launch the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web App:

1. **Set Option Parameters** (left sidebar):
   - Underlying Price (e.g., 4000 for SPX)
   - Strike Price (e.g., 4100)
   - Days to Expiration (e.g., 30)
   - Risk-Free Rate (e.g., 2%)
   - Option Type (Call or Put)

2. **Click "Calculate IV"** to get predictions

3. **Explore Visualizations**:
   - **Volatility Surface**: 3D surface plot showing IV across strikes and maturities
   - **Price Sensitivity**: Option price vs underlying price
   - **Model Comparison**: Compare ML model predictions

---

## Project Structure

```
FEC-European-options-7/
├── Notebooks/              # Jupyter notebooks for analysis
│   ├── model_training.ipynb
│   ├── stretch_goal_1_blackscholes.ipynb
│   └── stretch_goal_2_training_size.ipynb
│
├── data/                   # Data files
│   └── model_input/        # Preprocessed train/val/test data
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # Full documentation
```

---

## Key Features

### Machine Learning Models:
- **Random Forest**: Fast, interpretable tree ensemble
- **XGBoost**: State-of-the-art gradient boosting
- **Neural Networks**: Deep learning for complex patterns

### Performance:
- **Accuracy**: R² ≈ 0.41-0.42 on test set
- **Speed**: 100-1000x faster than numerical Black-Scholes
- **Training**: ~2M options from SPX 2022 data

---

## Example Use Cases

### 1. Quick IV Prediction
```python
# In the web app:
# Set: S=4000, K=4100, T=30 days, r=2%
# Get instant IV predictions from multiple models
```

### 2. Volatility Surface Analysis
```python
# Navigate to "Volatility Surface" tab
# View 3D surface across all strikes and maturities
```

### 3. Option Pricing
```python
# Get IV prediction
# App automatically calculates option price using Black-Scholes
```

---

## Troubleshooting

### Issue: Models not loading
**Solution**: Run the model training notebook first to generate model files.

### Issue: Import errors
**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Data files missing
**Solution**: The app includes demo models. For full functionality, ensure `data/model_input/` contains the preprocessed datasets.

### Issue: Streamlit not found
**Solution**: Install Streamlit:
```bash
pip install streamlit
```

---

## Performance Benchmarks

| Operation | Time |
|-----------|------|
| Single IV Prediction (ML) | ~0.01-0.05 ms |
| Single IV Prediction (Black-Scholes) | ~2-5 ms |
| Volatility Surface Generation | ~1-2 seconds |
| Model Training (full dataset) | ~2-5 minutes |

---

## Next Steps

1. **Explore the Notebooks**: Learn how models were trained and evaluated
2. **Try the Web App**: Get hands-on experience with IV prediction
3. **Read the README**: Understand the full project methodology
4. **Experiment**: Try different option parameters and analyze results

---

## Support

For questions or issues:
- Check the [README.md](README.md) for detailed documentation
- Review the Jupyter notebooks for implementation details
- Examine the stretch goal notebooks for advanced analysis

---

## Citation

If you use this project in your research or work, please cite:

```
European Options Implied Volatility Calculator
Machine Learning-based prediction using Random Forest, XGBoost, and Neural Networks
FEC-European-options-7 Team
2024
```

---

## License & Disclaimer

This project is for educational and research purposes only. Not financial advice.
Do not use for actual trading without proper validation and risk management.

---

**Happy Exploring! 📈**
