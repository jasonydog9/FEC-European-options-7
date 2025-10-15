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
  - ⏳ **Phase 2: Model Development:** In Progress
  - 📋 **Phase 3: Model Evaluation:** To Do
  - 📈 **Phase 4: Visualization & Analysis:** To Do
  - 🚀 **Phase 5: Documentation & Deployment:** To Do

-----

## 🧩 Tech Stack

  - **Language:** Python
  - **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/PyTorch
  - **Visualization:** Matplotlib, Seaborn
  - **Environment:** Jupyter Notebook / Google Colab

## 📂 Folder Structure

The project is organized into the following directory structure to keep data, notebooks, and outputs separate and manageable.

```
fec-european/
├── datasets/
│   ├── spx_eod_2022q1-ff0r18/
│   │   ├── spx_eod_202201.txt
│   │   └── ...
│   └── ... (all other raw quarterly data folders)
│
├── notebooks/
│   ├── data_processing.ipynb
│   └── ... (future modeling notebooks)
│
├── data/
│   ├── processed_by_file/
│   │   └── ... (cleaned, archived CSVs will be generated here)
│   └── model_input/
│       └── ... (final train/val/test CSVs will be generated here)
│
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

## 🚀 Stretch Goals
- **Comparison with Analytical Models:**  
  Benchmark ML models against the **Black-Scholes formula** to explore speed vs. accuracy trade-offs.

- **Training Data Sensitivity Analysis:**  
  Study the effect of **training set size** on model accuracy and runtime efficiency.

---

## 📈 Example Applications
- Real-time volatility estimation for SPX options.  
- Feature importance analysis to identify key market drivers.  
- Comparing classical financial models with ML-based pricing approaches.

---

