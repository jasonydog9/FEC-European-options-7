# Machine Learning for European Option Pricing (Team Name: FEC-European-options-7)

## 📘 Overview
This project explores the use of **machine learning models** to price **European-style options** and estimate **implied volatility** using open-source data from [OptionsDX.com](https://optionsdx.com/).  
The goal is to assess whether machine learning methods can match or outperform traditional numerical approaches (like Black-Scholes) in terms of both accuracy and computational efficiency.

---

## 🎯 Objectives
- Develop a **machine learning-based system** for pricing European options.  
- Analyze historical data to **predict market behavior** and implied volatility.  
- Evaluate and compare multiple algorithms for their predictive performance and interpretability.

---

## 📊 Project Status

  - ✅ **Phase 1: Data Collection & Preprocessing:** **Complete**
  - ✅ **Phase 2: Model Development:** **Complete**
  - ✅ **Phase 3: Model Evaluation:** **Complete**
  - ✅ **Phase 4: Visualization & Analysis:** **Complete**
  - ✅ **Phase 5: Documentation & Deployment:** **Complete**
  - ✅ **Stretch Goal 1: Black-Scholes Comparison:** **Complete**
  - ✅ **Stretch Goal 2: Training Set Size Analysis:** **Complete**

-----

## 🧩 Tech Stack

  - **Language:** Python
  - **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
  - **Visualization:** Matplotlib, Seaborn, Plotly
  - **Web Framework:** Streamlit
  - **Environment:** Jupyter Notebook / VS Code

## 📂 Folder Structure

The project is organized into the following directory structure to keep data, notebooks, and outputs separate and manageable.

```
FEC-European-options-7/
├── datasets/
│   └── spx_eod_*/           # Raw quarterly data files (not in repo)
│
├── Notebooks/
│   ├── data_collection.ipynb
│   ├── data_processing.ipynb
│   ├── model_training.ipynb
│   ├── test_random_split.ipynb
│   ├── stretch_goal_1_blackscholes.ipynb   # NEW: Black-Scholes comparison
│   └── stretch_goal_2_training_size.ipynb  # NEW: Training set analysis
│
├── data/
│   ├── processed_by_file/   # Cleaned CSVs
│   └── model_input/         # Train/val/test splits
│       ├── X_train.csv, y_train.csv
│       ├── X_val.csv, y_val.csv
│       └── X_test.csv, y_test.csv
│
├── models/                  # Saved model files (optional)
│
├── app.py                   # NEW: Streamlit web application
├── requirements.txt         # NEW: Python dependencies
└── README.md
```


## ⚙️ Key Steps

### 1. Data Collection
- Gather historical European option data (e.g., SPX index options).  
- Include variables such as **strike price**, **expiration date**, **underlying asset price**, and other relevant features.

⚠️ Raw datasets are not included due to GitHub size limits (several GBs). Download the data from Google Drive and place it in the correct local directory as described below.

### Step 1: Download the Datasets

1.  Download the zipped quarterly data folders from the Google Drive links provided in the **`Dataset Links`** document.
2.  Unzip all folders.
3.  Place all the unzipped `spx_eod_*` folders (e.g., `spx_eod_2022q1-ff0r18`, etc.) directly inside the **`datasets/`** directory at the project root.

### Step 2: Run the Preprocessing Pipeline

The entire data workflow is handled by a single, modular Jupyter Notebook.

1.  Navigate to the `notebooks/` folder in your local repository.
2.  Open **`1_data_processing_pipeline.ipynb`** in a Jupyter environment (like Jupyter Lab or VS Code).
3.  Execute all cells in the notebook from top to bottom by selecting "Run All".

### Step 3: Understand the Outputs

The pipeline will automatically create a new `data/` folder in the project root containing two sets of outputs:

  - **Archived Clean Data (`data/processed_by_file/`):** This folder contains a clean `.csv` file for each raw input `.txt` file, preserving the original quarterly folder structure. This is useful for archival purposes or ad-hoc analysis on specific months.

  - **Modeling Datasets (`data/model_input/`):** This folder contains the final, chronologically-split datasets that are ready for model training. This is the data you will use in all subsequent modeling notebooks.

      - `X_train.csv`, `y_train.csv`
      - `X_val.csv`, `y_val.csv`
      - `X_test.csv`, `y_test.csv`


### 2. Model Development
Implement and compare three models:
- **Random Forest** – Tree-based model for interpretability and feature importance.  
- **XGBoost** – Gradient-boosted ensemble model for structured data.  
- **Neural Networks (MLP)** – Captures complex non-linear relationships in pricing data.

### 3. Model Evaluation
Assess performance using:
- **RMSE** (Root Mean Squared Error)  
- **R² Score** (Coefficient of Determination)  
- **Training/Inference Time**

### 4. Visualization
- Plot **predicted vs. actual implied volatility surfaces**.  
- Visualize **feature importance** and **residual distributions**.

### 5. Documentation & Deployment
- Document findings, methodologies, and analysis results.  
- Optionally deploy via **Google Colab** or a **web-based interface** for real-time prediction.

---

## 📊 Expected Outcomes
- A **robust ML framework** for European option pricing written in Python.  
- **Insights** into the comparative performance of Random Forest, XGBoost, and Neural Networks.  
- A **deployable volatility estimation tool** for traders and risk managers.

---

## 🚀 Stretch Goals & Results

### ✅ Stretch Goal 1: ML vs Black-Scholes Comparison

**Research Questions:**
1. Is there a **time advantage** to using ML vs numerical methods?
2. Is there a **loss of accuracy** when using ML?

**Key Findings:**
- **Speed**: ML models are **100-1000x faster** than Black-Scholes Newton-Raphson solver
  - Black-Scholes: ~2-5 ms per option
  - Random Forest: ~0.01-0.05 ms per option
  - XGBoost: ~0.005-0.02 ms per option

- **Accuracy**: ML models match or **exceed** Black-Scholes accuracy
  - ML learns market microstructure patterns that Black-Scholes assumes away
  - Test R² scores: 0.40-0.45 on real market data

- **Conclusion**: For high-frequency trading and bulk calculations, **ML is superior** in both speed and accuracy

See [stretch_goal_1_blackscholes.ipynb](Notebooks/stretch_goal_1_blackscholes.ipynb) for detailed analysis.

---

### ✅ Stretch Goal 2: Training Set Size Analysis

**Research Questions:**
1. What is the **loss/gain in accuracy** due to training set size?
2. What is the **trade-off in training time**?
3. What is the **minimum viable training set size**?

**Key Findings:**
- **Accuracy**: More data improves accuracy, but with **diminishing returns**
  - 10% data → 25% data: Large improvement
  - 75% data → 100% data: Minimal improvement

- **Training Time**: Scales roughly **linearly** with dataset size
  - Random Forest: Most sensitive to data size
  - XGBoost: Best balance of speed and data efficiency

- **Optimal Size**: **50-75% of data** achieves 95%+ of maximum accuracy
  - Recommendation: Use 75% for production, 25-50% for development/testing

See [stretch_goal_2_training_size.ipynb](Notebooks/stretch_goal_2_training_size.ipynb) for detailed learning curves.

---

## 🌐 Web Application Deployment

A **Streamlit web application** has been developed for real-time implied volatility prediction!

### Features:
- Interactive parameter inputs (strike, underlying, expiration, etc.)
- Real-time IV predictions from multiple ML models
- 3D volatility surface visualization
- Option price sensitivity analysis (Greeks)
- Model performance comparison

### Running the App:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:8501`
   - The app will automatically reload when you update the code

### App Screenshot:
The web interface provides:
- Sidebar for option parameter inputs
- Real-time IV calculations from RF and XGBoost
- Interactive 3D volatility surface
- Price sensitivity charts
- Model comparison metrics

---

## 📊 Final Model Performance Summary

| Model | Test RMSE | Test R² | Training Time | Inference Speed |
|-------|-----------|---------|---------------|-----------------|
| Random Forest | 0.0436 | 0.414 | ~107s | ~0.02 ms/option |
| XGBoost | 0.0435 | 0.413 | ~8s | ~0.01 ms/option |
| Neural Network | 0.0438 | 0.412 | ~304s | ~0.05 ms/option |
| Black-Scholes | 0.0440 | 0.410 | N/A | ~2-5 ms/option |

**Best Overall Model**: **XGBoost** - Best balance of speed, accuracy, and training efficiency

---

## 📈 Example Applications
- Real-time volatility estimation for SPX options
- High-frequency trading applications requiring fast IV calculations
- Feature importance analysis to identify key market drivers
- Comparing classical financial models with ML-based pricing approaches
- Risk management and portfolio optimization

---

