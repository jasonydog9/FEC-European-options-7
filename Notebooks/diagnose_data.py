#!/usr/bin/env python3
"""
Diagnostic script to understand why R² is negative
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import glob

print("="*70)
print("DIAGNOSTIC: Understanding Negative R² Issue")
print("="*70)

# Load ALL processed files
print("\n1. Loading processed data files...")
processed_files = glob.glob('data/processed_by_file/**/*.csv', recursive=True)
print(f"   Found {len(processed_files)} files")

# Load first 5 files to test
df_list = []
for file in sorted(processed_files)[:5]:
    df = pd.read_csv(file, parse_dates=['QUOTE_DATE', 'EXPIRE_DATE'])
    df_list.append(df)
    print(f"   Loaded {file.split('/')[-1]}: {len(df):,} rows")

df_all = pd.concat(df_list, ignore_index=True)
df_all = df_all.sort_values('QUOTE_DATE').reset_index(drop=True)

print(f"\n   Total rows: {len(df_all):,}")
print(f"   Columns: {list(df_all.columns)}")

# Check date range
print(f"\n   Date range: {df_all['QUOTE_DATE'].min()} to {df_all['QUOTE_DATE'].max()}")

# Define features
feature_cols = ['T_years', 'moneyness', 'risk_free_rate',
                'strike_distance', 'strike_distance_pct', 'option_type_encoded']
target_col = 'iv'

# Check if all features exist
missing = [f for f in feature_cols if f not in df_all.columns]
if missing:
    print(f"\n   ⚠️  Missing features: {missing}")
    feature_cols = [f for f in feature_cols if f in df_all.columns]
    print(f"   Using available features: {feature_cols}")

# Split chronologically
train_size = int(len(df_all) * 0.70)
val_size = int(len(df_all) * 0.15)

df_train = df_all.iloc[:train_size]
df_val = df_all.iloc[train_size:train_size + val_size]
df_test = df_all.iloc[train_size + val_size:]

print("\n" + "="*70)
print("2. Data Split Analysis")
print("="*70)

for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f"\n{name} Set:")
    print(f"  Size: {len(df):,} rows")
    print(f"  Date range: {df['QUOTE_DATE'].min()} to {df['QUOTE_DATE'].max()}")
    print(f"  IV statistics:")
    print(f"    Mean:  {df[target_col].mean():.6f}")
    print(f"    Std:   {df[target_col].std():.6f}")
    print(f"    Var:   {df[target_col].var():.6f}")
    print(f"    Min:   {df[target_col].min():.6f}")
    print(f"    Max:   {df[target_col].max():.6f}")

# Extract features and target
X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

X_val = df_val[feature_cols].values
y_val = df_val[target_col].values

X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

# Check the variance issue
train_var = y_train.var()
val_var = y_val.var()
test_var = y_test.var()

print("\n" + "="*70)
print("3. TARGET VARIANCE ANALYSIS - KEY TO R² ISSUE!")
print("="*70)
print(f"\nTrain variance: {train_var:.8f}")
print(f"Val variance:   {val_var:.8f}  (ratio to train: {val_var/train_var:.2f}x)")
print(f"Test variance:  {test_var:.8f}  (ratio to train: {test_var/train_var:.2f}x)")

if val_var > 1.5 * train_var or test_var > 1.5 * train_var:
    print("\n⚠️  PROBLEM FOUND: Validation/Test variance >> Training variance")
    print("   This means the test data is from a different volatility regime!")
    print("   Even a perfect model on training data will get negative R² on test.")

# Try simple baseline
print("\n" + "="*70)
print("4. Baseline Performance (Predicting Mean)")
print("="*70)

train_mean = y_train.mean()
baseline_train_r2 = r2_score(y_train, np.full_like(y_train, train_mean))
baseline_val_r2 = r2_score(y_val, np.full_like(y_val, train_mean))
baseline_test_r2 = r2_score(y_test, np.full_like(y_test, train_mean))

print(f"\nPredicting train mean ({train_mean:.6f}) for all samples:")
print(f"  Train R²: {baseline_train_r2:.6f}")
print(f"  Val R²:   {baseline_val_r2:.6f}")
print(f"  Test R²:  {baseline_test_r2:.6f}")

if baseline_val_r2 < 0:
    print("\n⚠️  Even the baseline gets negative R²!")
    print("   This confirms train/val/test are from different distributions.")

# Try with scaling
print("\n" + "="*70)
print("5. Training Simple Random Forest WITH Scaling")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures after scaling (train):")
print(f"  Means: {X_train_scaled.mean(axis=0)}")
print(f"  Stds:  {X_train_scaled.std(axis=0)}")

# Train simple model
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
print(f"\nTraining Random Forest...")
rf.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = rf.predict(X_train_scaled)
y_val_pred = rf.predict(X_val_scaled)
y_test_pred = rf.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nRandom Forest Results:")
print(f"  Train R²: {train_r2:.6f}")
print(f"  Val R²:   {val_r2:.6f}  {'✅ POSITIVE!' if val_r2 > 0 else '❌ NEGATIVE'}")
print(f"  Test R²:  {test_r2:.6f}  {'✅ POSITIVE!' if test_r2 > 0 else '❌ NEGATIVE'}")

print(f"\n  Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.6f}")
print(f"  Val RMSE:   {np.sqrt(mean_squared_error(y_val, y_val_pred)):.6f}")
print(f"  Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_test_pred)):.6f}")

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importances:")
for _, row in importances.iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

print("\n" + "="*70)
print("6. RECOMMENDATIONS")
print("="*70)

if val_r2 < 0 or test_r2 < 0:
    print("\n❌ R² is still negative. Root cause:")
    print("   Your chronological split creates train/test from different volatility regimes.")
    print("\n   Solutions:")
    print("   1. Use RANDOM split instead of chronological split")
    print("   2. Use only more recent data (2023 only) for all splits")
    print("   3. Add time-based features to capture volatility regime changes")
    print("   4. Use a rolling window training approach")
    print("   5. Consider domain-specific models (GARCH, SVI for vol surface)")
else:
    print("\n✅ R² is positive! The fix worked.")
    print("   The model is learning meaningful patterns.")

print("\n" + "="*70)
