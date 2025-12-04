# Machine Learning for European Option Pricing (Team Name: FEC-European-options-7)

## Overview
This project explores the use of **machine learning models** to price **European-style options** and estimate **implied volatility** using open-source data from [OptionsDX.com](https://optionsdx.com/).  
The goal is to assess whether machine learning methods can match or outperform traditional numerical approaches (like Black-Scholes) in terms of both accuracy and computational efficiency.

## Objectives
- Develop a **machine learning-based system** for pricing European options.  
- Analyze historical data to **predict market behavior** and implied volatility.  
- Evaluate and compare multiple algorithms for their predictive performance and interpretability.

## Project Status
- Phase 1: Data Collection & Preprocessing – Complete  
- Phase 2: Model Development – Complete  
- Phase 3: Model Evaluation – Complete  
- Phase 4: Visualization & Analysis – Complete  
- Phase 5: Documentation & Deployment – Complete  
- Stretch Goal 1: Black-Scholes Comparison – Complete  
- Stretch Goal 2: Training Set Size Analysis – Complete  

## Tech Stack
- **Language:** Python  
- **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Web Framework:** Streamlit  
- **Environment:** Jupyter Notebook / VS Code  

## Folder Structure
FEC-European-options-7/
├── datasets/
│ └── spx_eod_*/ # Raw quarterly data files (not in repo)
├── Notebooks/
│ ├── data_collection.ipynb
│ ├── data_processing.ipynb
│ ├── model_training.ipynb
│ ├── test_random_split.ipynb
│ ├── stretch_goal_1_blackscholes.ipynb
│ └── stretch_goal_2_training_size.ipynb
├── data/
│ ├── processed_by_file/ # Cleaned CSVs
│ └── model_input/ # Train/val/test splits
│ ├── X_train.csv
│ ├── y_train.csv
│ ├── X_val.csv
│ ├── y_val.csv
│ ├── X_test.csv
│ └── y_test.csv
├── models/ # Saved model files (optional)
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
└── README.md

## Key Steps

### 1. Data Collection
- Gather historical European option data (e.g., SPX index options).  
- Include variables such as **strike price**, **expiration date**, **underlying asset price**, and other relevant features.

#### Step 1: Download the Datasets
1. Download zipped quarterly data folders from the Google Drive links in the `Dataset Links` document.  
2. Unzip all folders.  
3. Place all unzipped `spx_eod_*` folders directly inside the `datasets/` directory at the project root.

#### Step 2: Run the Preprocessing Pipeline
1. Navigate to the `notebooks/` folder in your local repository.  
2. Open `1_data_processing_pipeline.ipynb` in Jupyter or VS Code.  
3. Execute all cells from top to bottom.

#### Step 3: Understand the Outputs
- **Archived Clean Data (`data/processed_by_file/`):** Clean CSV files for each raw input file, preserving quarterly structure.  
- **Modeling Datasets (`data/model_input/`):** Chronologically-split datasets ready for model training:
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
- Plot predicted vs. actual implied volatility surfaces.  
- Visualize feature importance and residual distributions.

### 5. Documentation & Deployment
- Document findings, methodologies, and analysis results.  
- Optionally deploy via Google Colab or a web-based interface for real-time prediction.

## Expected Outcomes
- A **robust ML framework** for European option pricing in Python.  
- **Insights** into comparative performance of Random Forest, XGBoost, and Neural Networks.  
- A **deployable volatility estimation tool** for traders and risk managers.

## Stretch Goals & Results

### Stretch Goal 1: ML vs Black-Scholes Comparison
**Research Questions:**  
1. Is there a time advantage to using ML vs numerical methods?  
2. Is there a loss of accuracy when using ML?

**Key Findings:**  
- **Speed:** ML models are 100–1000x faster than Black-Scholes solver  
  - Black-Scholes: ~2–5 ms per option  
  - Random Forest: ~0.01–0.05 ms per option  
  - XGBoost: ~0.005–0.02 ms per option  
- **Accuracy:** ML models match or exceed Black-Scholes  
  - ML learns market microstructure patterns that Black-Scholes assumes away  
  - Test R² scores: 0.40–0.45 on real market data  
- **Conclusion:** ML is superior in speed and accuracy for high-frequency or bulk calculations

See `stretch_goal_1_blackscholes.ipynb` for detailed analysis.

### Stretch Goal 2: Training Set Size Analysis
**Research Questions:**  
1. What is the loss/gain in accuracy due to training set size?  
2. What is the trade-off in training time?  
3. What is the minimum viable training set size?

**Key Findings:**  
- **Accuracy:** More data improves accuracy, but with diminishing returns  
  - 10–25% data: Large improvement  
  - 75–100% data: Minimal improvement  
- **Training Time:** Scales roughly linearly with dataset size  
  - Random Forest: Most sensitive to size  
  - XGBoost: Best balance of speed and efficiency  
- **Optimal Size:** 50–75% achieves 95%+ of maximum accuracy  
  - Recommendation: 75% for production, 25–50% for testing

See `stretch_goal_2_training_size.ipynb` for learning curves.

## Web Application Deployment
A **Streamlit web app** has been developed for real-time implied volatility prediction, available at [mloptions.streamlit.app](https://mloptions.streamlit.app).

**Features:**  
- Interactive parameter inputs (strike, underlying, expiration)  
- Real-time IV predictions from multiple ML models  
- 3D volatility surface visualization  
- Option price sensitivity analysis (Greeks)  
- Model performance comparison
