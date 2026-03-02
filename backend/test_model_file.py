import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "cardio_model_gb_optimized.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    exit(1)

print(f"Loading {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 20 features matching training order
cols = [
    'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active', 'age_years', 'bmi',
    'smoke_age', 'smoke_bp', 'smoke_chol',
    'alco_age', 'alco_bp', 'alco_chol',
    'inactive_bmi', 'bp_ratio'
]

def make_input(gender=2, height=170, weight=80.0, ap_hi=130, ap_lo=85,
               cholesterol=1, gluc=1, smoke=0, alco=0, active=1, age_years=50.0):
    bmi = weight / ((height / 100) ** 2)
    return [
        gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
        smoke, alco, active, age_years, bmi,
        smoke * age_years,      # smoke_age
        smoke * ap_hi,          # smoke_bp
        smoke * cholesterol,    # smoke_chol
        alco * age_years,       # alco_age
        alco * ap_hi,           # alco_bp
        alco * cholesterol,     # alco_chol
        (1 - active) * bmi,     # inactive_bmi
        ap_hi / (ap_lo if ap_lo != 0 else 1)  # bp_ratio
    ]

def predict(input_data, label):
    df = pd.DataFrame([input_data], columns=cols)
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    print(f"  {label}: prediction={pred}, probability={prob:.4f}")
    return prob

print("\n=== Comparison Tests ===")

# Baseline: healthy non-smoker, non-drinker, active
p_base = predict(make_input(), "Baseline (no smoke, no alco, active)")

# Smoker only
p_smoke = predict(make_input(smoke=1), "Smoker only")

# Alcohol only
p_alco = predict(make_input(alco=1), "Alcohol only")

# Inactive only
p_inactive = predict(make_input(active=0), "Inactive only")

# Smoker + Alcohol + Inactive
p_all = predict(make_input(smoke=1, alco=1, active=0), "Smoker + Alcohol + Inactive")

print("\n=== Results ===")
all_pass = True

tests = [
    (p_smoke > p_base, "Smoker > Baseline"),
    (p_alco > p_base, "Alcohol > Baseline"),
    (p_inactive > p_base, "Inactive > Baseline"),
    (p_all > p_base, "All risk > Baseline"),
]

for passed, desc in tests:
    status = "[PASS]" if passed else "[FAIL]"
    if not passed:
        all_pass = False
    print(f"  {status} {desc}")

print(f"\n{'All tests PASSED!' if all_pass else 'Some tests FAILED!'}")
