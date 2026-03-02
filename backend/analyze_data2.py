import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('cardio_train_cleaned.csv')

results = []
results.append(f"Shape: {df.shape}")
results.append(f"Columns: {list(df.columns)}")
results.append("")

# Rate of cardio disease by smoking
results.append("=== Cardio rate by smoke ===")
for smoke_val in sorted(df['smoke'].unique()):
    rate = df[df['smoke']==smoke_val]['cardio'].mean()
    count = len(df[df['smoke']==smoke_val])
    results.append(f"  smoke={smoke_val}: cardio_rate={rate:.4f}, count={count}")

results.append("")
results.append("=== Cardio rate by active ===")
for active_val in sorted(df['active'].unique()):
    rate = df[df['active']==active_val]['cardio'].mean()
    count = len(df[df['active']==active_val])
    results.append(f"  active={active_val}: cardio_rate={rate:.4f}, count={count}")

results.append("")
results.append("=== Feature correlations with cardio ===")
corr = df.corr(numeric_only=True)['cardio'].drop('cardio').sort_values(ascending=False)
for feat, val in corr.items():
    results.append(f"  {feat}: {val:.4f}")

# Check model feature importances
results.append("")
results.append("=== Model Feature Importances ===")
model = joblib.load('cardio_model_gb_optimized.pkl')
cols = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi']
importances = model.feature_importances_
for col, imp in sorted(zip(cols, importances), key=lambda x: -x[1]):
    results.append(f"  {col}: {imp:.4f}")

# Test predictions: smoker vs non-smoker (all else equal)
results.append("")
results.append("=== Testing: smoker vs non-smoker (same params) ===")
scaler = joblib.load('scaler.pkl')

# Base person: 50 year old male, 170cm, 80kg, slightly elevated BP, normal cholesterol/glucose, no alcohol, active
base = {'gender': 2, 'height': 170, 'weight': 80.0, 'ap_hi': 130, 'ap_lo': 85,
        'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1,
        'age_years': 50.0, 'bmi': 80/(1.70**2)}

# Non-smoker, Active
non_smoker = pd.DataFrame([base], columns=cols)
scaled_ns = scaler.transform(non_smoker)
prob_ns = model.predict_proba(scaled_ns)[0][1]
results.append(f"  Non-smoker, Active:     prob={prob_ns:.4f}")

# Smoker, Active
smoker = base.copy()
smoker['smoke'] = 1
smoker_df = pd.DataFrame([smoker], columns=cols)
scaled_s = scaler.transform(smoker_df)
prob_s = model.predict_proba(scaled_s)[0][1]
results.append(f"  Smoker, Active:         prob={prob_s:.4f}")

# Non-smoker, Inactive
inactive = base.copy()
inactive['active'] = 0
inactive_df = pd.DataFrame([inactive], columns=cols)
scaled_i = scaler.transform(inactive_df)
prob_i = model.predict_proba(scaled_i)[0][1]
results.append(f"  Non-smoker, Inactive:   prob={prob_i:.4f}")

# Smoker, Inactive
smoker_inactive = base.copy()
smoker_inactive['smoke'] = 1
smoker_inactive['active'] = 0
si_df = pd.DataFrame([smoker_inactive], columns=cols)
scaled_si = scaler.transform(si_df)
prob_si = model.predict_proba(scaled_si)[0][1]
results.append(f"  Smoker, Inactive:       prob={prob_si:.4f}")

with open('analysis_results.txt', 'w') as f:
    f.write('\n'.join(results))

print("Results written to analysis_results.txt")
